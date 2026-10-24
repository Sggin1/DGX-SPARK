# File: test_memguard_events.py
# Location: tests/test_memguard_events.py
# Purpose: memory-guard line patterns (from mem_guard.py), fail-soft reads, report section.
# Dependencies: flightrec.memguard_events, flightrec.report, flightrec.cli

"""Canned journal text through the injectable runner. mem_guard.py is read, never run."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from flightrec import kernel_events as ke
from flightrec import memguard_events as mg
from flightrec.report import _fmt_memguard

P = "2026-01-02T03:04:{:02d}+00:00 testhost python[42]: "
JOURNAL = "\n".join([
    "2026-01-02T03:00:00+00:00 testhost systemd[1]: Started flightrec mem_guard.",
    P.format(1) + "[03:04:01] guard up: kill<20.0GB-avail warn>90.0GB-used (total=121) "
                  "poll=0.05s (20Hz) pid=42",
    P.format(5) + "[03:04:05] WARN used=95.2GB > 90.0GB (avail=25.8GB)",
    P.format(9) + "[03:04:09] BREACH avail=18.4GB < 20.0GB — killing runaways",
    P.format(9) + "  SIGKILL 1234: python3 -m vllm.entrypoints --model testmodel",
    P.format(10) + "  docker stop abc123 (/testcontainer)",
    "2026-01-02T03:05:00+00:00 testhost systemd[1]: flightrec-memguard.service: "
    "Main process exited, code=killed, status=9/KILL",
    "2026-01-02T03:05:01+00:00 testhost systemd[1]: Stopped flightrec mem_guard.",
])


def _runner(out="", err="", code=0, seen=None):
    def run(args, timeout):
        if seen is not None:
            seen.append(args)
        return code, out, err
    return run


def test_patterns_match_the_strings_mem_guard_prints():
    src = (Path(__file__).resolve().parents[1] / "flightrec" / "mem_guard.py").read_text(
        encoding="utf-8")
    # the f-string shapes this module relies on are still present in the source
    for fragment in ("BREACH avail=", "WARN used=", "SIGKILL {pid}:", "docker stop {cid} ("):
        assert fragment in src
    assert mg.classify("[03:04:05] WARN used=95.2GB > 90.0GB (avail=25.8GB)") == "warn"
    assert mg.classify("[03:04:09] BREACH avail=18.4GB < 20 — killing runaways") == "breach"
    assert mg.classify("  SIGKILL 1234: python3 -m vllm") == "kill"
    assert mg.classify("  docker stop abc123 (/name)") == "docker_stop"


def test_start_stop_and_startup_lines_ignored():
    events = mg.parse(JOURNAL)
    assert [e["category"] for e in events] == ["warn", "breach", "kill", "docker_stop"]
    assert events[0] == {"time": "2026-01-02T03:04:05+00:00", "category": "warn",
                         "line": "[03:04:05] WARN used=95.2GB > 90.0GB (avail=25.8GB)"}
    assert all("guard up" not in e["line"] and "Stopped" not in e["line"] for e in events)
    assert mg.classify("guard up: kill<20GB-avail") is None
    assert mg.classify("Started flightrec mem_guard.") is None


def test_scan_arguments_and_results():
    seen = []
    events, reason = mg.scan("a", "b", runner=_runner(JOURNAL, seen=seen))
    assert reason is None and len(events) == 4
    args = seen[0]
    assert args[:3] == ["journalctl", "-u", "flightrec-memguard"]
    assert args[args.index("--since") + 1] == "a" and args[args.index("--until") + 1] == "b"
    assert mg.scan(runner=_runner("-- No entries --\n")) == ([], None)


def test_unreadable_journal_reasons():
    msg = "No journal files were opened due to insufficient permissions."
    assert mg.scan(runner=_runner(err=msg)) == ([], "journal not readable (permission)")
    events, reason = mg.scan(runner=_runner(err="boom", code=1))
    assert events == [] and "exit 1" in reason

    def missing(args, timeout):
        raise FileNotFoundError("journalctl")

    def slow(args, timeout):
        raise subprocess.TimeoutExpired(args, timeout)

    assert mg.scan(runner=missing) == ([], "journalctl not found")
    assert "timed out" in mg.scan(runner=slow, timeout=2)[1]


def test_capture_for_run(tmp_path):
    info = mg.capture_for_run(tmp_path, 1000.0, 1010.0, runner=_runner(JOURNAL))
    assert info["status"] == "ok" and info["count"] == 4 and info["reason"] is None
    assert len(json.loads((tmp_path / "memguard_events.json").read_text())) == 4
    empty = tmp_path / "empty"
    empty.mkdir()
    info = mg.capture_for_run(empty, 1000.0, 1010.0, runner=_runner(""))
    assert info["status"] == "ok" and info["count"] == 0
    assert not (empty / "memguard_events.json").exists()
    info = mg.capture_for_run(empty, 1000.0, 1010.0, runner=_runner(err="x", code=1))
    assert info["status"] == "unreadable" and info["count"] == 0


def test_capture_for_run_never_raises(tmp_path):
    info = mg.capture_for_run(tmp_path, None, None)
    assert info["status"] == "unreadable" and info["count"] == 0


def test_report_section_none_some_unreadable_and_old_artifact():
    ok0 = {"memguard": {"status": "ok", "reason": None, "count": 0}}
    assert _fmt_memguard(ok0, []).strip() == "memory guard during the run: none"
    events = mg.parse(JOURNAL)
    ok4 = {"memguard": {"status": "ok", "reason": None, "count": 4}}
    text = _fmt_memguard(ok4, events, show=2)
    assert "memory guard during the run: 4 events" in text and "not marked invalid" in text
    assert "1 x [warn]" in text and "1 x [kill]" in text and "2 more groups" in text
    bad = {"memguard": {"status": "unreadable", "reason": "journalctl not found"}}
    assert "could not read (journalctl not found)" in _fmt_memguard(bad, [])
    assert _fmt_memguard({}, []) == ""


def _cli_main():
    try:  # cli imports the recorder, which needs time.CLOCK_MONOTONIC_RAW (Linux)
        from flightrec.cli import main
    except AttributeError:
        pytest.skip("flightrec.cli needs time.CLOCK_MONOTONIC_RAW")
    return main


def test_cli_guard_flag(monkeypatch, capsys):
    main = _cli_main()

    def fake(args, timeout):
        return (0, JOURNAL, "") if "-u" in args else (0, "", "")

    monkeypatch.setattr(ke, "_run", fake)
    assert main(["kernel-events", "--guard"]) == 0
    out = capsys.readouterr().out
    assert "memory guard: 4 events (listed only)" in out and "[breach]" in out
    assert main(["kernel-events", "--guard", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["memguard"]["events"]) == 4
    assert main(["kernel-events"]) == 0
    assert "memory guard" not in capsys.readouterr().out


def test_mem_guard_source_untouched_by_this_module():
    assert not re.search(r"memguard_events|platform_snapshot",
                         (Path(__file__).resolve().parents[1] / "flightrec"
                          / "mem_guard.py").read_text(encoding="utf-8"))


def test_recorder_flush_stamps_memguard_and_platform(monkeypatch, tmp_path):
    try:  # the recorder module needs time.CLOCK_MONOTONIC_RAW (Linux)
        from flightrec import recorder
    except AttributeError:
        pytest.skip("flightrec.recorder needs time.CLOCK_MONOTONIC_RAW")
    monkeypatch.setattr(recorder, "capture_header", lambda gpu_index: {})
    monkeypatch.setattr(ke, "_run", lambda args, timeout: (0, JOURNAL, ""))
    rec = recorder.FlightRecorder(tmp_path)
    rec._wall = [1000.0, 1010.0]
    rec._flush()
    header = json.loads((tmp_path / "header.json").read_text())
    assert header["memguard"]["status"] == "ok" and header["memguard"]["count"] == 4
    assert (tmp_path / "memguard_events.json").exists()
    assert "kernel_release" in header["platform"]
