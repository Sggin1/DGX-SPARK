# File: report.py
# Location: flightrec/report.py
# Purpose: Load a recorder artifact and print provenance + validity + rollup.
# Dependencies: pandas, flightrec.validate

"""Human-readable summary of one flight-recorder artifact."""

import json
from pathlib import Path

import pandas as pd

from flightrec.validate import verdict
from flightrec.phase_tree import build_tree, format_tree


def load_run(run_dir):
    """Load an artifact dir -> (samples_df, phases_df, header_dict)."""
    base = Path(run_dir)
    samples = pd.read_parquet(base / "samples.parquet")
    phases = pd.read_parquet(base / "phases.parquet")
    header = json.loads((base / "header.json").read_text(encoding="utf-8"))
    return samples, phases, header


def in_phases(samples, phases):
    """Samples inside any marked phase window (the active benchmark span)."""
    if phases.empty or "t0_ns" not in phases.columns:
        return samples
    mask = pd.Series(False, index=samples.index)
    for _, row in phases.iterrows():
        mask |= samples["t_ns"].between(row["t0_ns"], row["t1_ns"])
    return samples[mask]


def summarize_run(run_dir):
    """Print header + validity verdict + energy/clock rollup; return the verdict."""
    samples, phases, header = load_run(run_dir)
    active = in_phases(samples, phases)
    scoped = active if len(active) else samples
    result = verdict(scoped)
    print(_fmt_header(header))
    print(_fmt_verdict(result, scoped=len(active) > 0))
    print(_fmt_rollup(scoped))
    efficiency = _fmt_work_per_joule(work_per_joule(header, scoped, result))
    if efficiency:
        print(efficiency)
    kernel = _fmt_kernel_events(header, _load_kernel_events(run_dir))
    if kernel:
        print(kernel)
    tree = build_tree(phases, samples)
    if tree:
        print("\nphase tree (duration · energy · mean clock):\n" + format_tree(tree))
    return result


def work_per_joule(header, samples, result):
    """Work units per joule over the sampled window, from the GPU energy counter.

    ``header["work"]`` is the raw {units, unit} the run recorded; ``result`` is the
    validity verdict for ``samples``. Returns None when the run recorded no work,
    else a dict with ``status`` ok / invalid / no_energy and, only when ok,
    work_per_j and j_per_work. No number is given for an invalid run or when the
    energy counter is missing or did not advance.
    """
    work = header.get("work")
    if not work or not work.get("units"):
        return None
    out = {"status": "ok", "units": work["units"], "unit": work.get("unit", "units")}
    energy = samples["energy_mj"].dropna() if "energy_mj" in samples.columns else []
    if not result.get("valid"):
        out["status"] = "invalid"
    elif len(energy) < 2 or energy.max() <= energy.min():
        out["status"] = "no_energy"
    else:
        joules = float(energy.max() - energy.min()) / 1000.0
        out.update(energy_j=joules, work_per_j=work["units"] / joules,
                   j_per_work=joules / work["units"])
    return out


def _fmt_work_per_joule(w):
    """Lines for the work/J result, or "" when the run recorded no work."""
    if w is None:
        return ""
    head = f"\nwork: {w['units']:g} {w['unit']}"
    if w["status"] == "invalid":
        return head + " | work per joule: not computed (run INVALID, energy not trusted)"
    if w["status"] == "no_energy":
        return head + " | work per joule: not computed (no usable energy counter in window)"
    return (f"{head} | {w['work_per_j']:.4g} {w['unit']}/J | {w['j_per_work']:.4g} J/{w['unit']}"
            f" over {w['energy_j']:.1f} J\n"
            "  energy = GPU-reported counter (NVML), not wall power: excludes CPU, board, "
            "PSU losses")


def _load_kernel_events(run_dir):
    """Events stored by `record` (kernel_events.json), or [] when absent or unreadable."""
    try:
        return json.loads((Path(run_dir) / "kernel_events.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _fmt_kernel_events(header, events, show=5):
    """Kernel-log section; "" for older artifacts that have no ``kernel_events`` field."""
    info = header.get("kernel_events")
    if not info:
        return ""
    head = "\nkernel events during the run: "
    if info.get("status") != "ok":
        return head + f"could not read kernel log ({info.get('reason') or 'unknown reason'})"
    if not info.get("count"):
        return head + "none"
    counts = {}
    for e in events:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    by_cat = ", ".join(f"{c} {n}" for c, n in sorted(counts.items()))
    lines = [head + f"{info['count']}" + (f" ({by_cat})" if by_cat else "")]
    lines += [f"  {e['time']} [{e['category']}] {e['line']}" for e in events[:show]]
    if len(events) > show:
        lines.append(f"  ... {len(events) - show} more in kernel_events.json")
    return "\n".join(lines)


def _fmt_header(h):
    lines = [f"\n=== {h.get('gpu_name')}  driver {h.get('driver')} ===",
             f"SM clock ceiling {h.get('sm_clock_max_mhz')} MHz | "
             f"BW wall {h.get('bandwidth_wall_gbps')} GB/s "
             f"[{h.get('bandwidth_wall_source')}]"]
    if h.get("bandwidth_read_only_gbps"):
        lines.append(f"read-only BW {h['bandwidth_read_only_gbps']} GB/s "
                     "(separate from the triad wall; reads only)")
    peaks = h.get("compute_peak_tflops")
    if peaks:
        lines.append("compute peak " + ", ".join(f"{dt}={tf} TF" for dt, tf in peaks.items()))
    q = h.get("quiescence")
    if q:
        lines.append(f"pre-flight: {q['verdict']} ({'; '.join(q['reasons'])})")
    lines.append(f"caveat: {h.get('caveats')}")
    return "\n".join(lines)


def _fmt_verdict(v, scoped):
    tag = _verdict_tag(v)
    where = "in-phase" if scoped else "whole-run (no phases marked)"
    return (f"\nverdict [{where}]: {tag}\n"
            f"  samples={v.get('samples')} loaded={v.get('load_pct_of_window')}% | "
            f"throttled-of-loaded={v.get('throttled_pct_of_loaded')}% | "
            f"clock min/med={v.get('sm_clock_min_mhz')}/{v.get('sm_clock_median_mhz')} MHz | "
            f"power {v.get('power_min_w')}-{v.get('power_max_w')} W")


def _verdict_tag(v):
    if not v.get("loaded_samples"):
        return "VALID (GPU idle in window — nothing to judge)"
    return "VALID (under load, no throttle)" if v.get("valid") else "INVALID (throttled under load)"


def _fmt_rollup(samples):
    span = samples["t_rel_s"]
    dur = float(span.max() - span.min())
    joules = float(samples["energy_mj"].max() - samples["energy_mj"].min()) / 1000.0
    cores = int(samples["n_active"].max()) if "n_active" in samples.columns else 0
    return (f"\nrollup: {dur:.1f}s | mean power {samples['power_w'].mean():.1f} W | "
            f"GPU energy {joules:.1f} J | peak active cores {cores}/20")
