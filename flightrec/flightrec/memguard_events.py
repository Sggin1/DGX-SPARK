# File: memguard_events.py
# Location: flightrec/memguard_events.py
# Purpose: Read-only listing of memory-guard warn/kill lines from the systemd journal.
# Dependencies: journalctl (optional; absent or unreadable -> empty result plus a reason)

"""Memory-guard actions, listed as data.

``flightrec.mem_guard`` prints these messages (quoted from ``mem_guard.py``)::

    [HH:MM:SS] WARN used=<x>GB > <y>GB (avail=<z>GB)
    [HH:MM:SS] BREACH avail=<x>GB < <y>GB -- killing runaways
      SIGKILL <pid>: <cmdline[:90]>
      docker stop <id> (<name>)

This module only reads ``journalctl -u flightrec-memguard`` and keeps lines of those
shapes. Start-up ("guard up: ...") lines and systemd's own start/stop lines are ignored.
Nothing here changes or signals the guard, and no run is marked invalid by what it finds.
Note: the guard writes to its ``--log`` file; lines reach the journal only if the unit
sends that output there.
"""

import json
import re

from flightrec import kernel_events as ke

UNIT = "flightrec-memguard"
# `-o short-iso`: "<time> <host> <ident>[pid]: <message>"
_LINE = re.compile(r"^(\S+)\s+\S+\s+([^\s:\[]+)(?:\[\d+\])?: (.*)$")
_RULES = [
    ("breach", re.compile(r"^\[\d\d:\d\d:\d\d\] BREACH avail=")),
    ("warn", re.compile(r"^\[\d\d:\d\d:\d\d\] WARN used=")),
    ("kill", re.compile(r"^\s*SIGKILL \d+: ")),
    ("docker_stop", re.compile(r"^\s*docker stop \S+ \(")),
]


def classify(message):
    """Category for one guard message, or None (start-up and other lines are skipped)."""
    for name, pattern in _RULES:
        if pattern.search(message):
            return name
    return None


def group(events):
    """Counted stack of guard events (all digit runs are normalized: the lines are numeric)."""
    return ke.group(events, min_digits=1)


def parse(text):
    """Matching lines of ``journalctl -u ... -o short-iso`` output -> event dicts."""
    events = []
    for raw in text.splitlines():
        found = _LINE.match(raw)
        if not found or found.group(2) == "systemd":
            continue
        category = classify(found.group(3))
        if category:
            events.append({"time": found.group(1), "category": category,
                           "line": found.group(3).strip()})
    return events


def scan(since=None, until=None, boot=None, runner=None, timeout=ke.SCAN_TIMEOUT_S):
    """Guard events -> (events, reason); reason is None when the journal was read."""
    args = ["journalctl", "-u", UNIT, "--no-pager", "-o", "short-iso"]
    if boot is not None:
        args += ["-b", str(int(boot))]
    if since:
        args += ["--since", str(since)]
    if until:
        args += ["--until", str(until)]
    result, reason = ke._call(runner, args, timeout)
    if result is None:
        return [], reason
    code, out, err = result
    events = parse(out)
    if not events and ke._PERMISSION.search(err or ""):
        return [], "journal not readable (permission)"
    if code != 0 and not events:
        return [], f"journalctl exit {code}: {ke._first_line(err)}"
    return events, None


def capture_for_run(out_dir, start_unix, end_unix, runner=None):
    """Scan the run window; write ``memguard_events.json`` only if something matched.

    Returns the small record for ``header.json["memguard"]``. Never raises.
    """
    try:
        since, until = ke._stamp(start_unix - 1), ke._stamp(end_unix + 1)
        events, reason = scan(since, until, runner=runner)
        if events:
            (out_dir / "memguard_events.json").write_text(
                json.dumps(events[:ke.MAX_STORED_EVENTS], indent=2), encoding="utf-8")
        status = "ok" if reason is None else "unreadable"
        info = {"status": status, "reason": reason, "count": len(events),
                "since": since, "until": until}
        return ke._with_cap(info, len(events))
    except Exception as exc:  # noqa: BLE001 - must never break a recording
        return {"status": "unreadable", "reason": f"{exc.__class__.__name__}: {exc}",
                "count": 0, "since": None, "until": None}
