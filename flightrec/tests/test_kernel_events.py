# File: test_kernel_events.py
# Location: tests/test_kernel_events.py
# Purpose: kernel-log classification, fail-soft reads, multi-boot summary, report section.
# Dependencies: flightrec.kernel_events, flightrec.report, flightrec.cli

"""Canned journal text through the injectable ``runner``: no journalctl, no GPU."""

import json
import subprocess

import pytest

from flightrec import kernel_events as ke
from flightrec.report import _fmt_kernel_events

REFCNT = ("NVRM: refcntRequestReference_IMPL: Failed to enter state 1 "
          "(current state: 0, status: 0x00000056)")
JOURNAL = "\n".join([
    "2026-01-02T03:04:05+00:00 testhost kernel: " + REFCNT,
    "2026-01-02T03:04:06+00:00 testhost kernel: NVRM: Xid (PCI:000f:01:00): 79, fell off the bus",
    "2026-01-02T03:04:07+00:00 testhost kernel: audit: type=1400 msg=thermal oom NVRM:",
    "2026-01-02T03:04:08+00:00 testhost kernel: CPU temperature above threshold",
    "2026-01-02T03:04:09+00:00 testhost kernel: Out of memory: Killed process 1 (python)",
    "2026-01-02T03:04:10+00:00 testhost kernel: INFO: task foo:1 blocked, hung task detected",
    "2026-01-02T03:04:11+00:00 testhost kernel: usb 1-1: new device",
    "2026-01-02T03:04:12+00:00 testhost systemd[1]: not a kernel line, oom-kill",
])


def _runner(out="", err="", code=0, seen=None):
    def run(args, timeout):
        if seen is not None:
            seen.append(args)
        return code, out, err
    return run


def test_classification():
    assert ke.classify(REFCNT) == "nvrm"
    assert ke.classify("NVRM: Xid (PCI:0000:01:00): 31, pid=1") == "nvrm_xid"
    assert ke.classify("nvrm: xid 13") == "nvrm_xid"
    assert ke.classify("GPU slowdown engaged") == "throttle"
    assert ke.classify("oom-kill:constraint=CONSTRAINT_NONE") == "oom"
    assert ke.classify("soft lockup - CPU#3 stuck") == "watchdog"
    assert ke.classify("usb 1-1: new device") is None


def test_scan_keeps_matches_and_drops_audit_and_non_kernel_lines():
    events, reason = ke.scan(runner=_runner(JOURNAL))
    assert reason is None
    assert [e["category"] for e in events] == ["nvrm", "nvrm_xid", "thermal", "oom", "watchdog"]
    assert events[0] == {"time": "2026-01-02T03:04:05+00:00", "boot": 0,
                         "category": "nvrm", "line": REFCNT}


def test_scan_passes_window_and_boot_to_journalctl():
    seen = []
    ke.scan(since="1 hour ago", until="now", boot=-1, runner=_runner(seen=seen))
    args = seen[0]
    assert args[:2] == ["journalctl", "-k"]
    assert args[args.index("-b") + 1] == "-1"
    assert args[args.index("--since") + 1] == "1 hour ago"
    assert args[args.index("--until") + 1] == "now"


def test_permission_failure_returns_empty_plus_reason():
    msg = "No journal files were opened due to insufficient permissions."
    events, reason = ke.scan(runner=_runner(err=msg))
    assert events == [] and "permission" in reason
    events, reason = ke.scan(runner=_runner(err="boom", code=1))
    assert events == [] and "exit 1" in reason


def test_missing_journalctl_and_timeout_fail_soft():
    def missing(args, timeout):
        raise FileNotFoundError("journalctl")

    def slow(args, timeout):
        raise subprocess.TimeoutExpired(args, timeout)

    assert ke.scan(runner=missing) == ([], "journalctl not found")
    events, reason = ke.scan(runner=slow, timeout=3)
    assert events == [] and "timed out" in reason


BOOTS = (" -1 aaaa Fri 2026-01-01 00:00:00 UTC—Fri 2026-01-02 00:00:00 UTC\n"
         "  0 bbbb Fri 2026-01-02 00:01:00 UTC—Fri 2026-01-02 09:00:00 UTC\n")


def _multi_boot_runner(unreadable=()):
    def run(args, timeout):
        if "--list-boots" in args:
            return 0, BOOTS, ""
        boot = int(args[args.index("-b") + 1])
        if boot in unreadable:
            return 1, "", "Failed to read journal"
        return 0, f"2026-01-0{boot + 2}T00:00:00+00:00 testhost kernel: {REFCNT}\n", ""
    return run


def test_multi_boot_summary():
    events, failed = ke.scan_boots(runner=_multi_boot_runner())
    assert failed == []
    summary = ke.summarize(events)
    assert summary["total"] == 2
    assert summary["by_category"] == {"nvrm": 2}
    assert summary["by_boot"] == {-1: {"nvrm": 1}, 0: {"nvrm": 1}}


def test_multi_boot_names_unreadable_boots():
    events, failed = ke.scan_boots(runner=_multi_boot_runner(unreadable=(-1,)))
    assert [e["boot"] for e in events] == [0]
    assert failed[0][0] == -1 and "exit 1" in failed[0][1]


def test_list_boots_failure_is_reported():
    events, failed = ke.scan_boots(runner=_runner(err="denied", code=1))
    assert events == [] and failed[0][0] == "list-boots"


def test_capture_for_run_writes_file_only_when_matched(tmp_path):
    info = ke.capture_for_run(tmp_path, 1000.0, 1010.0, runner=_runner(JOURNAL))
    assert info["status"] == "ok" and info["count"] == 5 and info["reason"] is None
    assert len(json.loads((tmp_path / "kernel_events.json").read_text())) == 5
    empty = tmp_path / "empty"
    empty.mkdir()
    info = ke.capture_for_run(empty, 1000.0, 1010.0, runner=_runner(""))
    assert info["status"] == "ok" and info["count"] == 0
    assert not (empty / "kernel_events.json").exists()


def test_capture_for_run_never_raises(tmp_path):
    info = ke.capture_for_run(tmp_path, None, None)
    assert info["status"] == "unreadable" and info["count"] == 0


def test_report_section_none_some_unreadable_and_old_artifact():
    ok0 = {"kernel_events": {"status": "ok", "reason": None, "count": 0}}
    assert _fmt_kernel_events(ok0, []).strip() == "kernel events during the run: none"
    events = ke.parse(JOURNAL)
    ok5 = {"kernel_events": {"status": "ok", "reason": None, "count": 5}}
    text = _fmt_kernel_events(ok5, events, show=2)
    assert "kernel events during the run: 5 (nvrm 1, nvrm_xid 1, oom 1, thermal 1, watchdog 1)" in text
    assert REFCNT in text and "3 more" in text
    bad = {"kernel_events": {"status": "unreadable", "reason": "journal not readable (permission)"}}
    assert "could not read kernel log (journal not readable (permission))" in _fmt_kernel_events(bad, [])
    assert _fmt_kernel_events({}, []) == ""


def _cli_main():
    try:  # cli imports the recorder, which needs time.CLOCK_MONOTONIC_RAW (Linux)
        from flightrec.cli import main
    except AttributeError:
        pytest.skip("flightrec.cli needs time.CLOCK_MONOTONIC_RAW")
    return main


def test_cli_json_output(monkeypatch, capsys):
    main = _cli_main()
    monkeypatch.setattr(ke, "_run", lambda args, timeout: (0, JOURNAL, ""))
    assert main(["kernel-events", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["total"] == 5 and data["unreadable"] == []


def test_cli_table_caps_lines(monkeypatch, capsys):
    main = _cli_main()
    monkeypatch.setattr(ke, "_run", lambda args, timeout: (0, JOURNAL, ""))
    main(["kernel-events", "--max-lines", "2"])
    out = capsys.readouterr().out
    assert "kernel events: 5" in out and "nvrm_xid: 1" in out and "3 more" in out


def test_recorder_flush_stamps_kernel_events(monkeypatch, tmp_path):
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
    assert header["kernel_events"]["status"] == "ok"
    assert header["kernel_events"]["count"] == 5
    assert (tmp_path / "kernel_events.json").exists()


def test_boot_time_registration_lines_are_skipped_and_driver_oom_has_its_own_category():
    # lines seen on a healthy boot that earlier patterns counted as events
    for quiet in ("thermal_sys: Registered thermal governor 'step_wise'",
                  "acpi LNXTHERM:00: registered as thermal_zone0",
                  "ACPI: thermal: Thermal Zone [TSOC] (34 C)",
                  "NMI watchdog: Enabled. Permanently consumes one hw-PMU counter."):
        assert ke.classify(quiet) is None
    line = ("NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) "
            "returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1")
    assert ke.classify(line) == "nvrm_oom"
    assert ke.classify("Out of memory: Killed process 123 (python)") == "oom"
    assert ke.classify("INFO: task python:1 blocked for more than 120 seconds.") == "watchdog"
