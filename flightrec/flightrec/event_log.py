# File: event_log.py
# Location: flightrec/event_log.py
# Purpose: Optional rolling JSON-Lines log of counted event groups (size-capped, gzip backups).
# Dependencies: stdlib only (fcntl is used for a lock when present)

"""A small rolling event log, written only when asked for (``kernel-events --log``).

One JSON object per line, one record per group per scan::

    {ts, boot, category, key, example, count, first, last}      (+ host_label if given)

``append`` never raises: it returns ``(stats, reason)`` like the scanners here, with
``reason`` None on success. Nothing is scheduled or installed by this module.

Limits, as implemented:
  * a new record is merged into the file's final line (counts added, ``last`` updated)
    when it has the same boot and key and starts within ``dedupe_window_s`` of that
    line's ``last``; only the final line is considered;
  * when adding would pass ``max_bytes`` the file is rotated to ``<name>.1`` (gzip
    ``.1.gz`` when ``compress``), older backups shift up, and anything beyond
    ``backups`` is deleted;
  * ``max_age_days`` deletes rotated files whose modification time is older than that;
  * an advisory lock on ``<name>.lock`` is held while writing where ``fcntl`` exists.
"""

import gzip
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from flightrec import kernel_events as ke

try:  # advisory locking is POSIX-only; the module must still import elsewhere
    import fcntl
except ImportError:  # pragma: no cover - exercised by tests through monkeypatching
    fcntl = None

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUPS = 5
DEFAULT_DEDUPE_S = 60
TAIL_BYTES = 256 * 1024  # how much of the file end is read to find the final line


def default_path(environ=None):
    """``$XDG_STATE_HOME/flightrec/events.jsonl`` or ``~/.local/state/flightrec/events.jsonl``."""
    env = os.environ if environ is None else environ
    base = env.get("XDG_STATE_HOME") or ""
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".local" / "state"
    return root / "flightrec" / "events.jsonl"


def records_from_events(events, host_label=None, now=None):
    """Group ``events`` per boot -> log records (the ``ts`` is the scan time)."""
    ts = now or datetime.now().astimezone().isoformat(timespec="seconds")
    by_boot = {}
    for event in events:
        by_boot.setdefault(event.get("boot"), []).append(event)  # guard events have no boot
    records = []
    for boot, items in by_boot.items():
        for g in ke.group(items):
            record = {"ts": ts, "boot": boot, "category": g["category"], "key": g["key"],
                      "example": g["example"], "count": g["count"],
                      "first": g["first"], "last": g["last"]}
            if host_label:
                record["host_label"] = host_label
            records.append(record)
    return records


def append(records, path, max_bytes=DEFAULT_MAX_BYTES, backups=DEFAULT_BACKUPS,
           compress=True, max_age_days=None, dedupe_window_s=DEFAULT_DEDUPE_S):
    """Append records -> (stats, reason); ``stats`` = appended, merged, rotated, locked."""
    stats = {"appended": 0, "merged": 0, "rotated": False, "locked": False}
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(path, stats):
            for record in records:
                _add(record, path, max_bytes, backups, compress, dedupe_window_s, stats)
            if max_age_days is not None:
                _prune(path, max_age_days)
        return stats, None
    except Exception as exc:  # noqa: BLE001 - logging must never break the caller
        return stats, f"{exc.__class__.__name__}: {exc}"


class _locked:
    """Context manager: ``flock`` on ``<path>.lock`` if fcntl exists, else a no-op."""

    def __init__(self, path, stats):
        self.path, self.stats, self.handle = path, stats, None

    def __enter__(self):
        if fcntl is not None:
            try:
                self.handle = open(str(self.path) + ".lock", "a")
                fcntl.flock(self.handle, fcntl.LOCK_EX)
                self.stats["locked"] = True
            except OSError:
                self._release()
        return self

    def __exit__(self, *exc):
        self._release()
        return False

    def _release(self):
        if self.handle is not None:
            self.handle.close()  # closing drops the flock
            self.handle = None


def _add(record, path, max_bytes, backups, compress, window, stats):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    size = path.stat().st_size if path.exists() else 0
    if _merge_last(path, size, record, window):
        stats["merged"] += 1
        return
    if size and size + len(line.encode("utf-8")) > max_bytes:
        _rotate(path, backups, compress)
        stats["rotated"] = True
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
    stats["appended"] += 1


def _merge_last(path, size, record, window):
    """Fold ``record`` into the file's final line in place; False if it does not qualify."""
    if not size:
        return False
    with open(path, "rb") as handle:
        handle.seek(max(0, size - TAIL_BYTES))
        tail = handle.read()
    if not tail.endswith(b"\n"):
        return False
    cut = tail[:-1].rfind(b"\n") + 1
    if cut == 0 and size > len(tail):
        return False  # final line longer than the tail window: leave it alone
    try:
        prev = json.loads(tail[cut:].decode("utf-8"))
    except ValueError:
        return False
    if prev.get("key") != record["key"] or prev.get("boot") != record["boot"]:
        return False
    gap = _gap(prev.get("last"), record.get("first"))
    if gap is None or gap > window:
        return False
    prev["count"] = int(prev["count"]) + int(record["count"])
    if (_gap(prev["last"], record["last"]) or 0) > 0:
        prev["last"] = record["last"]
    prev["ts"] = record.get("ts", prev.get("ts"))
    with open(path, "r+b") as handle:
        handle.seek(size - len(tail) + cut)
        handle.truncate()
        handle.write((json.dumps(prev, ensure_ascii=False) + "\n").encode("utf-8"))
    return True


def _gap(earlier, later):
    """Seconds from ``earlier`` to ``later`` (negative if overlapping), or None."""
    a, b = ke.epoch(earlier), ke.epoch(later)
    return None if a is None or b is None else b - a


def _rotated(path, index):
    """Existing backup file number ``index`` (plain or .gz), or None."""
    for suffix in ("", ".gz"):
        candidate = Path(f"{path}.{index}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _rotate(path, backups, compress):
    if backups < 1:
        path.unlink()
        return
    for index in range(backups, 0, -1):
        found = _rotated(path, index)
        if found is None:
            continue
        if index >= backups:
            found.unlink()
        else:
            found.rename(Path(f"{path}.{index + 1}{'.gz' if found.suffix == '.gz' else ''}"))
    first = Path(f"{path}.1")
    if not compress:
        path.rename(first)
        return
    mtime = path.stat().st_mtime
    target = Path(f"{path}.1.gz")
    with open(path, "rb") as src, gzip.open(target, "wb") as dst:
        dst.write(src.read())
    os.utime(target, (mtime, mtime))
    path.unlink()


def _prune(path, max_age_days):
    cutoff = time.time() - float(max_age_days) * 86400.0
    for found in path.parent.glob(path.name + ".*"):
        if re.search(r"\.\d+(\.gz)?$", found.name) and found.stat().st_mtime < cutoff:
            found.unlink()


def read_all(path):
    """Records of the live file and its rotated backups, oldest first -> (records, skipped)."""
    path = Path(path)
    numbered = []
    for found in path.parent.glob(path.name + ".*"):
        hit = re.search(r"\.(\d+)(\.gz)?$", found.name)
        if hit and found.name == f"{path.name}.{hit.group(1)}{hit.group(2) or ''}":
            numbered.append((int(hit.group(1)), found))
    files = [f for _, f in sorted(numbered, key=lambda t: -t[0])]
    if path.exists():
        files.append(path)
    records, skipped = [], 0
    for found in files:
        try:
            opener = gzip.open if found.suffix == ".gz" else open
            with opener(found, "rt", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except (OSError, EOFError):
            skipped += 1
            continue
        for raw in text.splitlines():
            try:
                records.append(json.loads(raw))
            except ValueError:
                skipped += 1
    return records, skipped


def stack(records):
    """Fold log records into group dicts (counts summed per boot-independent key)."""
    merged = {}
    for r in records:
        slot = merged.setdefault((r["category"], r["key"]), {
            "key": r["key"], "example": r["example"], "category": r["category"],
            "count": 0, "first": r["first"], "last": r["last"], "by_boot": {}})
        slot["count"] += int(r["count"])
        if (_gap(slot["first"], r["first"]) or 0) < 0:
            slot["first"] = r["first"]
        if (_gap(slot["last"], r["last"]) or 0) > 0:
            slot["last"] = r["last"]
        boot = r.get("boot")
        slot["by_boot"][boot] = slot["by_boot"].get(boot, 0) + int(r["count"])
    return sorted(merged.values(), key=lambda g: (-g["count"], ke.epoch(g["first"]) or 0.0))


# --- --since-last state ---------------------------------------------------------------

def state_path(path):
    return Path(str(path) + ".state")


def load_state(path):
    """Saved ``{"last_event": <time string>}`` or {} when absent or unreadable."""
    try:
        data = json.loads(state_path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(path, **fields):
    """Merge ``fields`` (newest event times; None values are skipped) into the state file.

    Returns a reason string on failure, else None.
    """
    state = load_state(path)
    state.update({k: v for k, v in fields.items() if v})
    try:
        state_path(path).write_text(json.dumps(state), encoding="utf-8")
        return None
    except OSError as exc:
        return f"{exc.__class__.__name__}: {exc}"


def since_from_state(path, field="last_event"):
    """journalctl ``--since`` value (one second after the stored ``field`` time) or None."""
    stamp = ke.epoch(load_state(path).get(field))
    return None if stamp is None else ke._stamp(stamp + 1)


def latest_time(events):
    """Time string of the newest event, or None."""
    timed = [(ke.epoch(e["time"]), e["time"]) for e in events]
    timed = [t for t in timed if t[0] is not None]
    return max(timed)[1] if timed else None
