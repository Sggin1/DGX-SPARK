# File: test_work_per_joule.py
# Location: tests/test_work_per_joule.py
# Purpose: work-per-joule report math, its refusal cases, and the read-only BW calibration record.
# Dependencies: pandas, flightrec.report, flightrec.roofline

"""Synthetic frames and a mocked torch subprocess: no GPU needed.

Work/J is a report-layer ratio over the raw header work + the GPU energy counter; it must
refuse (no number) when energy is missing or the run was invalidated.
"""

import json
import subprocess

import pandas as pd

from flightrec import roofline
from flightrec.report import work_per_joule, _fmt_work_per_joule, summarize_run
from flightrec.validate import verdict

WORK = {"work": {"units": 1000, "unit": "tokens"}}


def _frame(clock=2300, power=60.0, energy=(0.0, 5000.0)):
    n = len(energy)
    data = {"t_ns": [i * 10**9 for i in range(n)], "t_rel_s": [float(i) for i in range(n)],
            "sm_clock_mhz": [clock] * n, "power_w": [power] * n, "gpu_busy_pct": [90.0] * n}
    if energy is not None:
        data["energy_mj"] = list(energy)
    return pd.DataFrame(data)


def _wpj(header, frame):
    return work_per_joule(header, frame, verdict(frame))


def test_work_per_joule_value():
    out = _wpj(WORK, _frame())  # 5000 mJ = 5 J for 1000 tokens
    assert out["status"] == "ok"
    assert out["work_per_j"] == 200.0
    assert out["j_per_work"] == 0.005
    assert out["unit"] == "tokens"


def test_no_work_recorded_is_none():
    assert _wpj({}, _frame()) is None
    assert _fmt_work_per_joule(None) == ""


def test_missing_energy_column_gives_no_number():
    frame = _frame().drop(columns=["energy_mj"])
    out = _wpj(WORK, frame)
    assert out["status"] == "no_energy"
    assert "work_per_j" not in out


def test_flat_or_single_energy_gives_no_number():
    assert _wpj(WORK, _frame(energy=(5000.0, 5000.0)))["status"] == "no_energy"
    assert _wpj(WORK, _frame(energy=(5000.0,)))["status"] == "no_energy"


def test_invalid_run_gives_no_number():
    wedge = _frame(clock=611, power=13.0)  # PD-wedge signature under load
    out = _wpj(WORK, wedge)
    assert out["status"] == "invalid"
    assert "work_per_j" not in out
    assert "not computed" in _fmt_work_per_joule(out)


def test_non_monotonic_energy_is_invalid():
    out = _wpj(WORK, _frame(energy=(0.0, 5000.0, 3000.0)))
    assert out["status"] == "invalid"


def test_output_states_gpu_counter_not_wall_power():
    text = _fmt_work_per_joule(_wpj(WORK, _frame()))
    assert "200 tokens/J" in text
    assert "not wall power" in text


def test_summarize_run_prints_work_per_joule(tmp_path, capsys):
    frame = _frame()
    frame["n_active"] = 4
    frame.to_parquet(tmp_path / "samples.parquet")
    pd.DataFrame([{"phase": None}]).to_parquet(tmp_path / "phases.parquet")
    (tmp_path / "header.json").write_text(json.dumps(WORK))
    summarize_run(str(tmp_path))
    assert "200 tokens/J" in capsys.readouterr().out


def test_calibrate_adds_read_only_and_keeps_triad_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    triad = {"bandwidth_wall_gbps_measured": 220.0, "method": "triad"}
    read = {"read_only_gbps_measured": 240.0, "read_only_method": "sum"}
    outputs = iter([triad, read])

    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess([], 0, json.dumps(next(outputs)), "")

    monkeypatch.setattr(roofline.subprocess, "run", fake_run)
    record = roofline.calibrate(host="testbox")
    saved = roofline.load_calibration(host="testbox")
    assert saved["bandwidth_wall_gbps_measured"] == 220.0
    assert saved["read_only_gbps_measured"] == 240.0
    assert "read_only_error" not in record


def test_calibrate_survives_read_only_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    results = iter([
        subprocess.CompletedProcess([], 0, json.dumps({"bandwidth_wall_gbps_measured": 220.0}), ""),
        subprocess.CompletedProcess([], 1, "", "CUDA out of memory"),
    ])
    monkeypatch.setattr(roofline.subprocess, "run", lambda *_a, **_k: next(results))
    roofline.calibrate(host="testbox")
    saved = roofline.load_calibration(host="testbox")
    assert saved["bandwidth_wall_gbps_measured"] == 220.0
    assert "read_only_gbps_measured" not in saved
    assert "out of memory" in saved["read_only_error"]
