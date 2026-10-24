# File: test_event_stack.py
# Location: tests/test_event_stack.py
# Purpose: counted event groups, the rolling event log, --group / --log / events-log show.
# Dependencies: flightrec.kernel_events, flightrec.event_log, flightrec.report, flightrec.cli

"""Canned journal text and tmp_path files only: no journalctl, no GPU, no sudo."""

import gzip
import json
import os
import time

import pytest

from flightrec import event_log as el
from flightrec import kernel_events as ke
from flightrec import memguard_events as mg
from flightrec.report import _fmt_kernel_events, _fmt_memguard

OOM = ("NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] "
       "(0x{code:08x}) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:{line}")


def _oom(second, code=0x51, line=1234, boot=0, minute=4):
    return {"time": f"2026-01-02T03:{minute:02d}:{second:02d}+00:00", "boot": boot,
            "category": "nvrm_oom", "line": OOM.format(code=code, line=line)}


def _journal(*events):
    return "\n".join(f"{e['time']} testhost kernel: {e['line']}" for e in events)


def _runner(text):
    return lambda args, timeout: (0, text, "")


def test_normalize_merges_numbers_and_keeps_different_messages_apart():
    a = ke.normalize(OOM.format(code=0x51, line=1234))
    b = ke.normalize(OOM.format(code=0x2a, line=987654))
    assert a == b and "<hex>" in a and "mem_desc.c:<n>" in a
    assert ke.normalize("NVRM: Xid (PCI:000f:01:00): 79, x") != ke.normalize(
        "NVRM: Xid (PCI:000f:01:00): 13, x")
    assert ke.normalize("NVRM: Xid (PCI:000f:01:00): 79") == ke.normalize(
        "NVRM: Xid (PCI:0001:0a:00): 79")
    assert ke.normalize("task python[4242]: blocked   at 2026-01-02T03:04:05+0000 now") == (
        "task python[<pid>]: blocked at <ts> now")
    assert ke.normalize("a 12:13:14 b [ 123.456] c") == "a <time> b [<uptime>] c"
    assert ke.normalize("Out of memory: Killed process 1 (python)") != ke.normalize(OOM.format(code=1, line=1))


def test_group_counts_first_last_burst_and_by_boot():
    events = [_oom(5, boot=-1), _oom(5, code=0x52, boot=-1), _oom(5, line=99999, boot=-1),
              _oom(9, boot=0), _oom(2, boot=0),
              {"time": "2026-01-02T03:04:07+00:00", "boot": 0, "category": "nvrm",
               "line": "NVRM: other thing 12"}]
    groups = ke.group(events)
    assert [g["count"] for g in groups] == [5, 1]
    top = groups[0]
    assert top["category"] == "nvrm_oom" and top["count"] == 5
    assert top["first"].endswith("03:04:02+00:00") and top["last"].endswith("03:04:09+00:00")
    assert top["example"] == OOM.format(code=0x51, line=1234)  # earliest event's raw line
    assert top["burst"] == 3 and top["by_boot"] == {-1: 3, 0: 2}
    assert groups[1]["burst"] == 1


def test_group_ties_sort_by_first_time():
    late = {"time": "2026-01-02T03:04:09+00:00", "boot": 0, "category": "nvrm", "line": "NVRM: b"}
    early = {"time": "2026-01-02T03:04:01+00:00", "boot": 0, "category": "nvrm", "line": "NVRM: a"}
    assert [g["example"] for g in ke.group([late, early])] == ["NVRM: a", "NVRM: b"]


def test_format_group_line():
    g = ke.group([_oom(5), _oom(5)])[0]
    text = ke.format_group(g)
    assert text.startswith("2 x [nvrm_oom] NVRM:") and "largest burst 2/s" in text
    assert "first 2026-01-02T03:04:05+00:00" in text


def _rec(key, count=1, first="2026-01-02T03:04:05+00:00", last=None, boot=0):
    return {"ts": "2026-01-02T04:00:00+00:00", "boot": boot, "category": "nvrm", "key": key,
            "example": key, "count": count, "first": first, "last": last or first}


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_append_writes_jsonl(tmp_path):
    path = tmp_path / "sub" / "events.jsonl"
    stats, reason = el.append([_rec("a"), _rec("b")], path)
    assert reason is None and stats["appended"] == 2
    assert [r["key"] for r in _lines(path)] == ["a", "b"]


def test_dedupe_window_merges_final_line_only_within_window(tmp_path):
    path = tmp_path / "events.jsonl"
    el.append([_rec("a", 3, "2026-01-02T03:04:05+00:00", "2026-01-02T03:04:10+00:00")], path)
    stats, _ = el.append([_rec("a", 2, "2026-01-02T03:04:40+00:00", "2026-01-02T03:04:50+00:00")],
                         path, dedupe_window_s=60)
    assert stats["merged"] == 1 and stats["appended"] == 0
    rows = _lines(path)
    assert len(rows) == 1 and rows[0]["count"] == 5
    assert rows[0]["first"].endswith("03:04:05+00:00") and rows[0]["last"].endswith("03:04:50+00:00")
    # outside the window: a new line
    stats, _ = el.append([_rec("a", 1, "2026-01-02T03:09:00+00:00")], path, dedupe_window_s=60)
    assert stats["appended"] == 1 and len(_lines(path)) == 2
    # different key never merges
    el.append([_rec("b", 1, "2026-01-02T03:09:01+00:00")], path)
    assert [r["key"] for r in _lines(path)] == ["a", "a", "b"]
    # different boot never merges
    el.append([_rec("b", 1, "2026-01-02T03:09:02+00:00", boot=-1)], path)
    assert len(_lines(path)) == 4


def test_rotation_backups_limit_and_gzip_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    for i in range(12):
        stats, reason = el.append([_rec(f"key{i}", first=f"2026-01-02T03:{i:02d}:00+00:00")],
                                  path, max_bytes=400, backups=3)
        assert reason is None
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "events.jsonl.1.gz" in names and "events.jsonl.3.gz" in names
    assert "events.jsonl.4.gz" not in names and "events.jsonl.4" not in names
    assert path.stat().st_size <= 400 + 400  # live file stays near the cap
    with gzip.open(tmp_path / "events.jsonl.1.gz", "rt", encoding="utf-8") as handle:
        newest_backup = [json.loads(x) for x in handle.read().splitlines()]
    assert newest_backup and newest_backup[0]["key"].startswith("key")
    records, skipped = el.read_all(path)
    assert skipped == 0
    keys = [r["key"] for r in records]
    assert keys == sorted(keys, key=lambda k: int(k[3:])) and keys[-1] == "key11"
    assert len(keys) < 12  # the oldest rotated out beyond `backups`


def test_rotation_without_compression_and_zero_backups(tmp_path):
    path = tmp_path / "events.jsonl"
    for i in range(4):
        el.append([_rec(f"k{i}")], path, max_bytes=300, backups=2, compress=False)
    names = {p.name for p in tmp_path.iterdir()}
    assert "events.jsonl.1" in names and not any(n.endswith(".gz") for n in names)
    other = tmp_path / "z" / "events.jsonl"
    for i in range(4):
        el.append([_rec(f"k{i}")], other, max_bytes=300, backups=0)
    assert [p.name for p in other.parent.iterdir() if p.name.startswith("events") and not p.name.endswith(".lock")] == ["events.jsonl"]


def test_max_age_prunes_rotated_files_by_mtime(tmp_path):
    path = tmp_path / "events.jsonl"
    old, fresh = tmp_path / "events.jsonl.2.gz", tmp_path / "events.jsonl.1.gz"
    for f in (old, fresh):
        f.write_bytes(gzip.compress(b"{}\n"))
    ten_days_ago = time.time() - 10 * 86400
    os.utime(old, (ten_days_ago, ten_days_ago))
    stats, reason = el.append([_rec("a")], path, max_age_days=5)
    assert reason is None and not old.exists() and fresh.exists() and path.exists()
    el.append([_rec("b")], path, max_age_days=None)
    assert fresh.exists()


def test_lock_fallback_when_fcntl_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(el, "fcntl", None)
    stats, reason = el.append([_rec("a")], tmp_path / "events.jsonl")
    assert reason is None and stats["locked"] is False and stats["appended"] == 1
    assert not (tmp_path / "events.jsonl.lock").exists()


def test_append_fails_soft_with_a_reason(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    stats, reason = el.append([_rec("a")], blocker / "events.jsonl")
    assert reason and stats["appended"] == 0


def test_default_path_follows_xdg_state_home(tmp_path):
    assert el.default_path({"XDG_STATE_HOME": str(tmp_path)}) == (
        tmp_path / "flightrec" / "events.jsonl")
    fallback = el.default_path({}).as_posix()
    assert fallback.endswith(".local/state/flightrec/events.jsonl")
    assert not fallback.startswith("/tmp")
    assert el.default_path({"XDG_STATE_HOME": "relative"}).as_posix().endswith(
        ".local/state/flightrec/events.jsonl")


def test_records_from_events_group_per_boot_and_optional_host_label():
    events = [_oom(5, boot=-1), _oom(6, boot=0), _oom(7, boot=0)]
    records = el.records_from_events(events, now="T")
    assert sorted((r["boot"], r["count"]) for r in records) == [(-1, 1), (0, 2)]
    assert all("host_label" not in r and r["ts"] == "T" for r in records)
    assert all(r["host_label"] == "testhost" for r in
               el.records_from_events(events, host_label="testhost"))


def test_state_roundtrip_and_since(tmp_path):
    path = tmp_path / "events.jsonl"
    assert el.since_from_state(path) is None
    assert el.save_state(path, last_event="2026-01-02T03:04:05+00:00") is None
    el.save_state(path, last_guard="2026-01-02T03:05:05+00:00", last_event=None)
    state = el.load_state(path)
    assert state["last_event"].startswith("2026-01-02T03:04:05") and "last_guard" in state
    since = el.since_from_state(path)
    stamp = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M:%S"))
    assert stamp == pytest.approx(ke.epoch("2026-01-02T03:04:05+00:00") + 1)
    assert el.latest_time([_oom(5), _oom(9), _oom(7)]).endswith("03:04:09+00:00")


def test_report_sections_show_top_groups_and_truncation_note():
    events = [_oom(5 + i % 3, code=i, line=1000 + i) for i in range(6)]
    events.append({"time": "2026-01-02T03:04:01+00:00", "boot": 0, "category": "nvrm", "line": "NVRM: odd"})
    header = {"kernel_events": {"status": "ok", "reason": None, "count": 7}}
    text = _fmt_kernel_events(header, events, show=1)
    assert "kernel events during the run: 7 (nvrm 1, nvrm_oom 6)" in text
    assert "6 x [nvrm_oom] NVRM: nvCheckOkFailedNoLog" in text and "largest burst 2/s" in text
    assert "1 more groups in kernel_events.json" in text and "NVRM: odd" not in text
    header["kernel_events"].update(count=9000, truncated=True, stored=2000)
    assert "holds the first 2000 of 9000 lines" in _fmt_kernel_events(header, events, show=1)
    guard = [{"time": f"2026-01-02T03:04:0{i}+00:00", "category": "warn",
              "line": f"[03:04:0{i}] WARN used={90 + i}.5GB > 90.0GB (avail={20 - i}.1GB)"}
             for i in range(3)]
    gtext = _fmt_memguard({"memguard": {"status": "ok", "reason": None, "count": 3}}, guard)
    assert "3 x [warn]" in gtext


def test_capture_caps_stored_lines_but_keeps_true_count(monkeypatch, tmp_path):
    monkeypatch.setattr(ke, "MAX_STORED_EVENTS", 5)
    events = [_oom(i % 50, code=i) for i in range(12)]
    info = ke.capture_for_run(tmp_path, 1000.0, 1010.0, runner=_runner(_journal(*events)))
    assert info["count"] == 12 and info["truncated"] is True and info["stored"] == 5
    assert len(json.loads((tmp_path / "kernel_events.json").read_text())) == 5
    small = tmp_path / "small"
    small.mkdir()
    info = ke.capture_for_run(small, 1000.0, 1010.0, runner=_runner(_journal(*events[:3])))
    assert "truncated" not in info


def _cli_main():
    try:  # cli imports the recorder, which needs time.CLOCK_MONOTONIC_RAW (Linux)
        from flightrec.cli import main
    except AttributeError:
        pytest.skip("flightrec.cli needs time.CLOCK_MONOTONIC_RAW")
    return main


def _patch_journal(monkeypatch, text, seen=None):
    def run(args, timeout):
        if seen is not None:
            seen.append(args)
        return (0, text, "") if "-k" in args else (0, "", "")
    monkeypatch.setattr(ke, "_run", run)


def test_cli_group_text_and_json(monkeypatch, capsys):
    main = _cli_main()
    events = [_oom(5, code=i, line=100 + i) for i in range(4)] + [_oom(9)]
    _patch_journal(monkeypatch, _journal(*events))
    assert main(["kernel-events", "--group"]) == 0
    out = capsys.readouterr().out
    assert "5 x [nvrm_oom]" in out and "largest burst 4/s" in out and "kernel events: 5" in out
    assert main(["kernel-events", "--group", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["groups"][0]["count"] == 5 and "events" not in data
    assert main(["kernel-events", "--group", "--max-groups", "0"]) == 0
    assert "1 more groups" in capsys.readouterr().out


def test_cli_group_with_all_boots_and_guard(monkeypatch, capsys):
    main = _cli_main()
    boots = (" -1 aaaa Fri 2026-01-01 00:00:00 UTC\n  0 bbbb Fri 2026-01-02 00:01:00 UTC\n")
    guard = ("2026-01-02T03:04:05+00:00 testhost python[42]: [03:04:05] WARN used=95.2GB > 90.0GB (avail=25.8GB)\n"
             "2026-01-02T03:04:06+00:00 testhost python[42]: [03:04:06] WARN used=96.2GB > 90.0GB (avail=24.8GB)\n")

    def run(args, timeout):
        if "--list-boots" in args:
            return 0, boots, ""
        if "-u" in args:
            return 0, guard, ""
        boot = int(args[args.index("-b") + 1])
        return 0, _journal(_oom(5, boot=boot)) + "\n", ""

    monkeypatch.setattr(ke, "_run", run)
    assert main(["kernel-events", "--all-boots", "--group", "--guard"]) == 0
    out = capsys.readouterr().out
    assert "2 x [nvrm_oom]" in out and "by boot: -1: 1, 0: 1" in out
    assert "memory guard: 2 events in 1 groups" in out and "2 x [warn]" in out


def test_cli_log_since_last_and_events_log_show(monkeypatch, capsys, tmp_path):
    main = _cli_main()
    log = tmp_path / "events.jsonl"
    seen = []
    _patch_journal(monkeypatch, _journal(_oom(5), _oom(5, code=2), _oom(8)), seen)
    assert main(["kernel-events", "--log", str(log), "--since-last", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "event log: 1 records appended, 0 merged" in out and "kernel events:" not in out
    assert _lines(log)[0]["count"] == 3
    assert "--since" not in seen[0]  # no state yet
    state = json.loads((tmp_path / "events.jsonl.state").read_text())
    assert state["last_event"].endswith("03:04:08+00:00")
    # second run: window starts after the stored time, and a new event there is added
    seen.clear()
    _patch_journal(monkeypatch, _journal(_oom(30)), seen)
    assert main(["kernel-events", "--log", str(log), "--since-last", "--quiet"]) == 0
    capsys.readouterr()
    assert "--since" in seen[0]
    rows = _lines(log)
    assert len(rows) == 1 and rows[0]["count"] == 4  # within 60 s of the previous last: merged

    assert main(["events-log", "show", str(log)]) == 0
    out = capsys.readouterr().out
    assert "1 records" in out and "4 x NVRM:" in out
    assert main(["events-log", "show", str(log), "--group", "--last", "5"]) == 0
    assert "4 x [nvrm_oom]" in capsys.readouterr().out


def test_cli_log_default_path_and_since_last_needs_log(monkeypatch, capsys, tmp_path):
    main = _cli_main()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    _patch_journal(monkeypatch, _journal(_oom(5)))
    assert main(["kernel-events", "--log"]) == 0
    assert (tmp_path / "flightrec" / "events.jsonl").exists()
    capsys.readouterr()
    assert main(["kernel-events", "--since-last"]) == 2
    assert "--since-last needs --log" in capsys.readouterr().out


def test_cli_log_failure_is_reported_not_raised(monkeypatch, capsys, tmp_path):
    main = _cli_main()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    _patch_journal(monkeypatch, _journal(_oom(5)))
    assert main(["kernel-events", "--log", str(blocker / "events.jsonl")]) == 1
    assert "event log: could not write" in capsys.readouterr().out


def test_mem_guard_source_not_touched():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "flightrec" / "mem_guard.py").read_text(
        encoding="utf-8")
    assert "event_log" not in text


def test_lock_is_taken_and_released_when_fcntl_exists(monkeypatch, tmp_path):
    calls = []

    class FakeFcntl:
        LOCK_EX = 2

        @staticmethod
        def flock(handle, mode):
            calls.append((handle.name.endswith(".lock"), mode))

    monkeypatch.setattr(el, "fcntl", FakeFcntl)
    stats, reason = el.append([_rec("a")], tmp_path / "events.jsonl")
    assert reason is None and stats["locked"] is True and calls == [(True, 2)]
