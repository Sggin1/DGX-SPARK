# File: test_platform_snapshot.py
# Location: tests/test_platform_snapshot.py
# Purpose: platform snapshot parsing, fail-soft fields, diff, report block, CLI.
# Dependencies: flightrec.platform_snapshot, flightrec.report, flightrec.cli

"""Fake ``root`` trees and canned journalctl output: no GPU, no journal, no sudo."""

import json

import pytest

from flightrec import platform_snapshot as ps
from flightrec.report import _fmt_platform

MEM_LINE = ("Memory: 117234560K/121634816K available (17408K kernel code, 4800K rwdata, "
            "14336K rodata, 8960K init, 1280K bss, 4400256K reserved, 0K cma-reserved)")
BOOT_LOG = "\n".join([
    "Booting Linux on physical CPU 0x0000000000 [0x410fd8e0]",
    "kernel: " + MEM_LINE,
    "arm_mpam: mpam: MPAM ACPI table not found",
    "mpam: second line is ignored",
    "usb 1-1: new device",
])
MEMINFO = ("MemTotal:       121635000 kB\nMemFree:         1000 kB\n"
           "MemAvailable:   117000000 kB\nCmaTotal:          65536 kB\n"
           "HugePages_Total:       0\nHugepagesize:       2048 kB\n")


def _tree(root, with_all=True):
    files = {
        "proc/sys/kernel/osrelease": "6.11.0-testkernel\n",
        "sys/class/dmi/id/bios_version": "1.2.3\n",
        "sys/class/dmi/id/bios_date": "09/30/2026\n",
        "sys/class/dmi/id/product_name": "TestBox\n",
        "proc/meminfo": MEMINFO,
        "proc/driver/nvidia/version":
            "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.95.05  Release Build\n"
            "GCC version:  gcc 13.3.0\n",
    }
    for rel, text in files.items():
        if with_all or rel == "proc/meminfo":
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if with_all:
        (root / "sys/fs/resctrl").mkdir(parents=True)
        for name in ("software", "arm_spe_0", "armv8_pmuv3_0"):
            (root / "sys/bus/event_source/devices" / name).mkdir(parents=True)


def _runner(out=BOOT_LOG, code=0, err="", seen=None):
    def run(args, timeout):
        if seen is not None:
            seen.append((args, timeout))
        return code, out, err
    return run


def test_every_field_parsed(tmp_path):
    _tree(tmp_path)
    seen = []
    snap = ps.snapshot(root=tmp_path, runner=_runner(seen=seen))
    assert snap["kernel_release"] == "6.11.0-testkernel"
    assert (snap["bios_version"], snap["bios_date"], snap["product_name"]) == (
        "1.2.3", "09/30/2026", "TestBox")
    assert snap["mem_total_kb"] == 121635000 and snap["mem_available_kb"] == 117000000
    assert snap["cma_total_kb"] == 65536 and snap["hugepages_total"] == 0
    assert snap["boot_memory_line"] == MEM_LINE
    assert snap["boot_memory"] == {"available_k": 117234560, "total_k": 121634816,
                                   "reserved_k": 4400256, "cma_reserved_k": 0}
    assert snap["mpam_line"] == "arm_mpam: mpam: MPAM ACPI table not found"
    assert snap["resctrl_present"] is True
    assert snap["perf_devices"] == ["arm_spe_0", "armv8_pmuv3_0", "software"]
    assert snap["nvidia_module_version"] == "580.95.05"
    assert snap["kernel_log_error"] is None
    args, timeout = seen[0]
    assert args[:3] == ["journalctl", "-k", "-b"] and args[3] == "0" and timeout > 0
    json.dumps(snap)


def test_missing_files_give_blank_fields_not_exceptions(tmp_path):
    _tree(tmp_path, with_all=False)
    snap = ps.snapshot(root=tmp_path, runner=_runner(out=""))
    assert snap["mem_total_kb"] == 121635000
    assert snap["kernel_release"] is None and snap["bios_version"] is None
    assert snap["cma_total_kb"] == 65536
    assert snap["perf_devices"] is None and snap["resctrl_present"] is False
    assert snap["nvidia_module_version"] is None
    assert snap["boot_memory_line"] is None and snap["mpam_line"] is None
    empty = ps.snapshot(root=tmp_path / "nowhere", runner=_runner(out=""))
    assert empty["mem_total_kb"] is None


def test_journal_failures_blank_only_the_log_fields(tmp_path):
    _tree(tmp_path)

    def missing(args, timeout):
        raise FileNotFoundError("journalctl")

    snap = ps.snapshot(root=tmp_path, runner=missing)
    assert snap["kernel_log_error"] == "journalctl not found"
    assert snap["boot_memory"] is None and snap["kernel_release"] == "6.11.0-testkernel"
    snap = ps.snapshot(root=tmp_path, runner=_runner(out="", code=1, err="denied"))
    assert "exit 1" in snap["kernel_log_error"]


def test_no_mpam_line_and_memory_line_with_journal_prefix(tmp_path):
    prefixed = "2026-01-02T03:04:05+00:00 testhost kernel: " + MEM_LINE
    snap = ps.snapshot(root=tmp_path, runner=_runner(out=prefixed))
    assert snap["boot_memory_line"] == MEM_LINE and snap["mpam_line"] is None


def test_diff_lists_only_differing_fields():
    a = {"kernel_release": "6.11", "bios_version": "1.0", "perf_devices": ["x"]}
    b = {"kernel_release": "6.14", "bios_version": "1.0", "perf_devices": ["x"], "new": 1}
    assert ps.diff(a, b) == {"kernel_release": ("6.11", "6.14"), "new": (None, 1)}
    assert ps.diff(a, a) == {}


def test_report_block_and_old_artifact(tmp_path):
    _tree(tmp_path)
    header = {"platform": ps.snapshot(root=tmp_path, runner=_runner())}
    text = _fmt_platform(header)
    assert "kernel 6.11.0-testkernel | BIOS 1.2.3 (09/30/2026)" in text
    assert "MemTotal 121635000 kB | boot reserved 4400256 kB" in text
    assert "MPAM: arm_mpam: mpam: MPAM ACPI table not found" in text
    assert "resctrl yes | perf devices 3" in text
    assert len(text.strip().splitlines()) <= 5
    blank = _fmt_platform({"platform": {"mem_total_kb": 5}})
    assert "MPAM: none seen" in blank and "resctrl no | perf devices unknown" in blank
    assert _fmt_platform({}) == ""


def _cli_main():
    try:  # cli imports the recorder, which needs time.CLOCK_MONOTONIC_RAW (Linux)
        from flightrec.cli import main
    except AttributeError:
        pytest.skip("flightrec.cli needs time.CLOCK_MONOTONIC_RAW")
    return main


def test_cli_platform_diff(tmp_path, capsys):
    main = _cli_main()
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps({"platform": {"kernel_release": "6.11", "bios_version": "1.0"}}))
    b.write_text(json.dumps({"kernel_release": "6.14", "bios_version": "1.0"}))
    c.write_text(json.dumps({"gpu_name": "old artifact"}))
    assert main(["platform-diff", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "kernel_release: 6.11 -> 6.14" in out and "bios_version" not in out
    assert main(["platform-diff", str(a), str(a)]) == 0
    assert capsys.readouterr().out.strip() == "no differences"
    assert main(["platform-diff", str(a), str(c)]) == 1
    assert main(["platform-diff", str(a), str(tmp_path / "missing.json")]) == 1


def test_cli_platform_json(monkeypatch, capsys):
    main = _cli_main()
    monkeypatch.setattr(ps, "snapshot", lambda: {"kernel_release": "6.11"})
    assert main(["platform", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"kernel_release": "6.11"}
