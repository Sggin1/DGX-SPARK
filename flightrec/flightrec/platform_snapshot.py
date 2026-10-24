# File: platform_snapshot.py
# Location: flightrec/platform_snapshot.py
# Purpose: Read-only snapshot of firmware/OS/memory facts, stored in each run header.
# Dependencies: stdlib only (/proc, /sys, journalctl optional)

"""Raw platform facts, so a firmware or OS change can be seen afterwards.

Each field is read on its own: a missing file or a failed command blanks that one field
(``None``) and nothing else. Values are stored as found; no judgement is made. Nothing
host-specific is read (no hostname, user, serial number, UUID).

    snap = snapshot()                               # real machine
    snap = snapshot(root=fake_dir, runner=canned)   # tests
    print(diff(snap_a, snap_b))
"""

import re
from pathlib import Path

from flightrec import kernel_events as ke

_MEMORY_LINE = re.compile(r"Memory: (\d+)K/(\d+)K available")
_RESERVED = re.compile(r"(\d+)K reserved")
_CMA_RESERVED = re.compile(r"(\d+)K cma-reserved")
_NVIDIA_VERSION = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
_MEMINFO = {"mem_total_kb": "MemTotal", "mem_available_kb": "MemAvailable",
            "cma_total_kb": "CmaTotal", "hugepages_total": "HugePages_Total"}


def snapshot(root="/", runner=None, timeout=ke.BOOT_TIMEOUT_S):
    """Dict of raw platform values; any unreadable field is None. Never raises."""
    base = Path(root)
    out = {
        "kernel_release": _text(base / "proc/sys/kernel/osrelease"),
        "bios_version": _text(base / "sys/class/dmi/id/bios_version"),
        "bios_date": _text(base / "sys/class/dmi/id/bios_date"),
        "product_name": _text(base / "sys/class/dmi/id/product_name"),
    }
    out.update(_meminfo(base / "proc/meminfo"))
    out.update(_boot_log(runner, timeout))
    out["resctrl_present"] = _guard(lambda: (base / "sys/fs/resctrl").exists(), False)
    out["perf_devices"] = _guard(
        lambda: sorted(p.name for p in (base / "sys/bus/event_source/devices").iterdir()))
    out.update(_nvidia(base / "proc/driver/nvidia/version"))
    return out


def parse_memory_line(line):
    """Numbers from the kernel's 'Memory: <avail>K/<total>K available (...)' line."""
    found = _MEMORY_LINE.search(line or "")
    if not found:
        return {}
    reserved, cma = _RESERVED.search(line), _CMA_RESERVED.search(line)
    return {"available_k": int(found.group(1)), "total_k": int(found.group(2)),
            "reserved_k": int(reserved.group(1)) if reserved else None,
            "cma_reserved_k": int(cma.group(1)) if cma else None}


def diff(a, b):
    """{field: (a_value, b_value)} for the fields that differ between two snapshots."""
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def _guard(thunk, default=None):
    try:
        return thunk()
    except Exception:  # noqa: BLE001 - one bad field must not blank the others
        return default


def _text(path):
    return _guard(lambda: path.read_text(encoding="utf-8", errors="replace").strip() or None)


def _meminfo(path):
    out = {key: None for key in _MEMINFO}
    for line in (_text(path) or "").splitlines():
        name, _, rest = line.partition(":")
        for key, label in _MEMINFO.items():
            if name == label and rest.split():
                out[key] = _guard(lambda: int(rest.split()[0]))
    return out


def _boot_log(runner, timeout):
    """Boot memory line + first 'mpam:' line from the current boot's kernel log."""
    out = {"boot_memory_line": None, "boot_memory": None, "mpam_line": None,
           "kernel_log_error": None}
    args = ["journalctl", "-k", "-b", "0", "--no-pager", "-o", "cat"]
    result, reason = ke._call(runner, args, timeout)
    if result is None:
        out["kernel_log_error"] = reason
        return out
    code, text, err = result
    if code != 0 and not text:
        out["kernel_log_error"] = f"journalctl exit {code}: {ke._first_line(err)}"
        return out
    for line in text.splitlines():
        if out["boot_memory_line"] is None and _MEMORY_LINE.search(line):
            out["boot_memory_line"] = line[line.index("Memory:"):].strip()
            out["boot_memory"] = parse_memory_line(line)
        if out["mpam_line"] is None and "mpam:" in line.lower():
            out["mpam_line"] = line.strip()
    return out


def _nvidia(path):
    first = (_text(path) or "").splitlines()[:1]
    version = _NVIDIA_VERSION.search(first[0]) if first else None
    return {"nvidia_module_version": version.group(1) if version else None}
