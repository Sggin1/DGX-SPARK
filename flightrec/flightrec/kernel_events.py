# File: kernel_events.py
# Location: flightrec/kernel_events.py
# Purpose: Read-only scan of the kernel log (systemd journal) for GPU/driver-related lines.
# Dependencies: journalctl (optional; absent or unreadable -> empty result plus a reason)

"""Kernel-side messages, listed as data.

Reads ``journalctl -k`` (the journal already keeps kernel messages across boots; nothing
is installed or enabled) and keeps lines that match a few broad patterns. The lines are
returned as found; no judgement is made about what any of them mean. Every reader here
fails soft: a missing journalctl, a permission problem or a timeout gives an empty list
and a short reason, never an exception.

    events, reason = scan(since="2 days ago")
    print(summarize(events))
"""

import json
import re
import subprocess
import time
from collections import Counter

# First match wins, so an "NVRM: Xid" line is nvrm_xid and not also nvrm.
_RULES = [
    ("nvrm_xid", re.compile(r"NVRM: Xid", re.I)),
    ("nvrm", re.compile(r"NVRM:", re.I)),
    ("thermal", re.compile(r"thermal|overheat", re.I)),
    ("throttle", re.compile(r"throttl|slowdown", re.I)),
    ("oom", re.compile(r"out of memory|oom-kill", re.I)),
    ("watchdog", re.compile(r"hung task|soft lockup|watchdog", re.I)),
]
_AUDIT = re.compile(r"^audit(\[\d+\])?:", re.I)
# `-o short-iso`: "<time> <host> kernel: <message>"
_LINE = re.compile(r"^(\S+)\s+\S+\s+kernel: (.*)$")
_PERMISSION = re.compile(r"permission|not seeing messages|No journal files", re.I)

CATEGORIES = [name for name, _ in _RULES]
SCAN_TIMEOUT_S = 15.0
BOOT_TIMEOUT_S = 20.0


def classify(message):
    """Category name for one kernel message, or None to skip it (incl. audit lines)."""
    if _AUDIT.match(message):
        return None
    for name, pattern in _RULES:
        if pattern.search(message):
            return name
    return None


def _run(args, timeout):
    """Default runner: (returncode, stdout, stderr) of a command."""
    done = subprocess.run(args, capture_output=True, text=True, errors="replace",
                          timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def _call(runner, args, timeout):
    """Run through ``runner``; return ((rc, out, err), None) or (None, reason)."""
    try:
        return (runner or _run)(args, timeout), None
    except FileNotFoundError:
        return None, "journalctl not found"
    except subprocess.TimeoutExpired:
        return None, f"journalctl timed out after {timeout:g}s"
    except OSError as exc:
        return None, f"journalctl failed to start ({exc.__class__.__name__})"


def parse(text, boot=0):
    """Matching lines of ``journalctl -k -o short-iso`` output -> list of event dicts."""
    events = []
    for raw in text.splitlines():
        found = _LINE.match(raw)
        category = classify(found.group(2)) if found else None
        if category:
            events.append({"time": found.group(1), "boot": boot,
                           "category": category, "line": found.group(2)})
    return events


def scan(since=None, until=None, boot=None, runner=None, timeout=SCAN_TIMEOUT_S):
    """Kernel-log events -> (events, reason); reason is None when the log was read.

    ``boot`` is a journalctl boot offset (0 = current, -1 = previous); default current.
    ``runner(args, timeout) -> (returncode, stdout, stderr)`` can be injected for tests.
    """
    boot_n = 0 if boot is None else int(boot)
    args = ["journalctl", "-k", "--no-pager", "-o", "short-iso", "-b", str(boot_n)]
    if since:
        args += ["--since", str(since)]
    if until:
        args += ["--until", str(until)]
    result, reason = _call(runner, args, timeout)
    if result is None:
        return [], reason
    code, out, err = result
    events = parse(out, boot_n)
    if not events and _PERMISSION.search(err or ""):
        return [], "journal not readable (permission)"
    if code != 0 and not events:
        return [], f"journalctl exit {code}: {_first_line(err)}"
    return events, None


def list_boots(runner=None, timeout=SCAN_TIMEOUT_S):
    """Boot offsets known to the journal -> (offsets, reason)."""
    result, reason = _call(runner, ["journalctl", "--list-boots", "--no-pager"], timeout)
    if result is None:
        return [], reason
    code, out, err = result
    offsets = [int(line.split()[0]) for line in out.splitlines()
               if re.match(r"^\s*-?\d+\s", line)]
    if not offsets:
        return [], (f"journalctl exit {code}: {_first_line(err)}" if code else "no boots listed")
    return offsets, None


def scan_boots(since=None, until=None, runner=None, boot_timeout=BOOT_TIMEOUT_S):
    """Scan every listed boot separately -> (events, failed); failed = [(boot, reason)]."""
    offsets, reason = list_boots(runner)
    if reason:
        return [], [("list-boots", reason)]
    events, failed = [], []
    for offset in offsets:
        found, why = scan(since, until, offset, runner, boot_timeout)
        events += found
        if why:
            failed.append((offset, why))
    return events, failed


def summarize(events):
    """Counts: {total, by_category, by_boot: {boot: {category: n}}}."""
    by_boot = {}
    for event in events:
        by_boot.setdefault(event["boot"], Counter())[event["category"]] += 1
    return {"total": len(events),
            "by_category": dict(Counter(e["category"] for e in events)),
            "by_boot": {boot: dict(c) for boot, c in sorted(by_boot.items())}}


def capture_for_run(out_dir, start_unix, end_unix, runner=None):
    """Scan the run window; write ``kernel_events.json`` only if something matched.

    Returns the small raw record for ``header.json["kernel_events"]``. Never raises.
    """
    try:
        since, until = _stamp(start_unix - 1), _stamp(end_unix + 1)
        events, reason = scan(since, until, runner=runner)
        if events:
            (out_dir / "kernel_events.json").write_text(
                json.dumps(events, indent=2), encoding="utf-8")
        status = "ok" if reason is None else "unreadable"
        return {"status": status, "reason": reason, "count": len(events),
                "since": since, "until": until}
    except Exception as exc:  # noqa: BLE001 - must never break a recording
        return {"status": "unreadable", "reason": f"{exc.__class__.__name__}: {exc}",
                "count": 0, "since": None, "until": None}


def _stamp(unix):
    """Local 'YYYY-MM-DD HH:MM:SS' for journalctl --since/--until."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix))


def _first_line(text):
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else "no message"
