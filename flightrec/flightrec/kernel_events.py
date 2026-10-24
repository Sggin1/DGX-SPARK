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
from datetime import datetime

# First match wins, so an "NVRM: Xid" line is nvrm_xid and not also nvrm.
_RULES = [
    ("nvrm_xid", re.compile(r"NVRM: Xid", re.I)),
    # driver-side allocation failure (seen on GB10 under memory pressure); kept apart from other NVRM lines
    ("nvrm_oom", re.compile(r"NVRM:.*(NV_ERR_NO_MEMORY|out of memory)", re.I)),
    ("nvrm", re.compile(r"NVRM:", re.I)),
    # warning-like wording only: boot-time "Registered thermal governor" / "thermal zone registered" lines are skipped
    ("thermal", re.compile(r"overheat|thermal (shutdown|trip|event)|critical temperature|"
                           r"temperature.*(above|exceed|too high)|thermal.*(critical|trip point)", re.I)),
    ("throttle", re.compile(r"throttl|slowdown", re.I)),
    ("oom", re.compile(r"out of memory|oom-kill|oom_reaper|invoked oom-killer", re.I)),
    # lockups and stalls; boot-time "NMI watchdog: Enabled" style lines are skipped
    ("watchdog", re.compile(r"hung task|soft lockup|hard lockup|blocked for more than|rcu.*stall|"
                            r"watchdog.*(bug|detected|expired|timeout)", re.I)),
]
_AUDIT = re.compile(r"^audit(\[\d+\])?:", re.I)
# `-o short-iso`: "<time> <host> kernel: <message>"
_LINE = re.compile(r"^(\S+)\s+\S+\s+kernel: (.*)$")
_PERMISSION = re.compile(r"permission|not seeing messages|No journal files", re.I)

CATEGORIES = [name for name, _ in _RULES]
SCAN_TIMEOUT_S = 15.0
BOOT_TIMEOUT_S = 20.0
# Stored raw lines per artifact file; a runaway log must not bloat a recording.
MAX_STORED_EVENTS = 2000

# Placeholders for normalize(); order matters (timestamps and PCI before bare digit runs).
_NORMALIZE = [
    (re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?"), "<ts>"),
    (re.compile(r"\[\s*\d+\.\d+\]"), "[<uptime>]"),
    (re.compile(r"\b\d\d:\d\d:\d\d(?:\.\d+)?\b"), "<time>"),
    (re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}(?:\.\d)?\b", re.I), "<pci>"),
    (re.compile(r"\b[0-9a-f]{2}:[0-9a-f]{2}\.\d\b", re.I), "<pci>"),
    (re.compile(r"\[\d+\]"), "[<pid>]"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<hex>"),
    (re.compile(r"(?<![\w.])\d+(?:\.\d+)+"), "<n>"),
]
_WS = re.compile(r"\s+")


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
                json.dumps(events[:MAX_STORED_EVENTS], indent=2), encoding="utf-8")
        status = "ok" if reason is None else "unreadable"
        info = {"status": status, "reason": reason, "count": len(events),
                "since": since, "until": until}
        return _with_cap(info, len(events))
    except Exception as exc:  # noqa: BLE001 - must never break a recording
        return {"status": "unreadable", "reason": f"{exc.__class__.__name__}: {exc}",
                "count": 0, "since": None, "until": None}


def _with_cap(info, total):
    """Add ``truncated``/``stored`` when the file holds fewer lines than ``count``."""
    if total > MAX_STORED_EVENTS:
        info.update(truncated=True, stored=MAX_STORED_EVENTS)
    return info


def _stamp(unix):
    """Local 'YYYY-MM-DD HH:MM:SS' for journalctl --since/--until."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix))


def _first_line(text):
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else "no message"


def normalize(message, min_digits=3):
    """Grouping key for a message: numbers and addresses become placeholders.

    Replaced by ``<ts>``, ``<time>``, ``[<uptime>]``, ``<pci>``, ``[<pid>]``, ``<hex>``
    (``0x...``), ``<n>`` (decimals, and digit runs of ``min_digits`` or more); whitespace
    is collapsed. Digit runs shorter than ``min_digits`` are kept (so "Xid 79" and
    "Xid 13" stay apart by default).
    """
    text = message
    for pattern, token in _NORMALIZE:
        text = pattern.sub(token, text)
    text = re.sub(r"\d{%d,}" % max(1, int(min_digits)), "<n>", text)
    return _WS.sub(" ", text).strip()


def epoch(text):
    """Seconds since the epoch for a journal ``short-iso`` time, or None if unparsable."""
    try:
        t = str(text).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        found = re.search(r"([+-])(\d\d)(\d\d)$", t)
        if found and ":" not in t[-5:]:
            t = t[:-5] + f"{found.group(1)}{found.group(2)}:{found.group(3)}"
        stamp = datetime.fromisoformat(t)
        return (stamp.astimezone() if stamp.tzinfo is None else stamp).timestamp()
    except (ValueError, TypeError):
        return None


def group(events, min_digits=3):
    """Counted stack of events -> list of group dicts.

    Each group: ``key``, ``example`` (first raw line, in time order), ``category``,
    ``count``, ``first``, ``last`` (time strings as found), ``burst`` (most occurrences
    inside one calendar second; 1 when times are unparsable) and ``by_boot``
    ({boot: n}, empty for events without a boot). Sorted by count (largest first),
    then by first time.
    """
    buckets = {}
    for index, event in enumerate(events):
        key = normalize(event["line"], min_digits)
        buckets.setdefault((event["category"], key), []).append((index, event))
    groups = []
    for (category, key), items in buckets.items():
        stamped = sorted(((epoch(e["time"]), i, e) for i, e in items),
                         key=lambda t: (t[0] is None, t[0] or 0.0, t[1]))
        seconds = Counter(int(t[0]) if t[0] is not None else str(t[2]["time"])[:19]
                          for t in stamped)
        boots = Counter(e["boot"] for e in (t[2] for t in stamped) if "boot" in e)
        groups.append({"key": key, "example": stamped[0][2]["line"], "category": category,
                       "count": len(stamped), "first": stamped[0][2]["time"],
                       "last": stamped[-1][2]["time"], "burst": max(seconds.values()),
                       "by_boot": dict(sorted(boots.items())),
                       "_sort": stamped[0][0] if stamped[0][0] is not None else 0.0})
    groups.sort(key=lambda g: (-g["count"], g["_sort"], g["key"]))
    for g in groups:
        del g["_sort"]
    return groups


def format_group(g):
    """One line: ``173 x [nvrm_oom] <example>  (first ..., last ..., largest burst N/s)``."""
    parts = [f"first {g['first']}", f"last {g['last']}"]
    if g.get("burst"):
        parts.append(f"largest burst {g['burst']}/s")
    return f"{g['count']} x [{g['category']}] {g['example']}  ({', '.join(parts)})"
