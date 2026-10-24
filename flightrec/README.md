# flightrec — GB10 flight recorder: run validity, memory guard and monitoring

Always-on, low-overhead machine-state telemetry for **NVIDIA DGX Spark (GB10 / sm_121a)** (earlier called "Tier-1" in these notes).

Three parts:

1. **Record and validity.** Wrap a bench or job and get a self-describing artifact: GPU power / clock / throttle / energy, CPU, and memory pressure on **one timeline**. Runs that throttled are marked **invalid**, which keeps power-path and thermal wedges out of results.
2. **Memory guard (`mem_guard`).** On unified-memory GB10, an out-of-memory condition often shows up as a frozen host rather than a clean error. `mem_guard` is an opt-in daemon that warns and, below a kill threshold, ends matching runaway processes (and stops non-registry containers). Review the flags before enabling it; a threshold that is too tight can end a job that would have finished.
3. **Monitoring.** `smoke` (pre-flight scan), `watch` (stall detection) and `live` (throttle alarm).

Benchmark statistics (adaptive-N sampling, gates, aggregation) are included for results that need them.

**Role:** data provider — clean, gated artifacts for analysis, not the analyst.

**Primary target:** DGX Spark / GB10 (validated). Core NVML + `/proc` paths may work on other Linux + NVIDIA hosts; validity, unified-memory semantics, and roofline defaults are Spark-tuned. Discrete-GPU / Intel host support is deferred.

Sibling: two-box fabric notes in [`../dual-spark/`](../dual-spark/).

---

## Install

Python **≥ 3.12**. From this directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# optional plots:
# uv pip install -e ".[dev,plot]"
```

Or: `pip install -e ".[dev]"`.

CLI entrypoint: `flightrec` (also `python -m flightrec`).

### Calibrate once per box (recommended)

Needs a torch+CUDA interpreter for STREAM-triad wall measurement:

```bash
flightrec calibrate --python /path/to/torch-venv/bin/python
# or: export FLIGHTREC_TORCH_PYTHON=...
```

Persists under `~/.cache/flightrec/calibration-<hostname>.json`.  
Calibration also records a read-only streaming number next to the triad (see [Read-only bandwidth](#read-only-bandwidth)).  
Example measured walls on GB10 LPDDR5X: **~220–226 GB/s** (~80–83% of the 273 GB/s spec). Uncalibrated fallback is a modeled **195 GB/s** constant.

---

## Why it exists (OOM / throttle)

On unified-memory GB10, “GPU OOM” often looks like **host freezes**, swap thrash, or silent stalls — not a clean CUDA OOM. Flightrec gives you:

| Arm | When | What |
|---|---|---|
| `record` + validity | every bench | clock+power droop invalidates wedged runs |
| `mem_guard` | long jobs / daemons | kill/warn on low `MemAvailable` |
| `smoke` | before multi-hour work | short slice → bottleneck + ETA |
| `watch` | during long jobs | stall = marker silent **and** GPU idle |
| `live` | interactive | live throttle alarm (same wedge rule as gate) |

Example unit file: [`systemd/flightrec-memguard.service`](systemd/flightrec-memguard.service) (edit user/paths).  
**Note:** on kill threshold, `mem_guard` can **SIGKILL** matching processes and `docker stop` non-`registry` containers — intentional for GB10 freeze recovery; review flags before enabling as a daemon.

---

## Command reference

| command | what it does | key flags (defaults) |
|---|---|---|
| `record` | wrap a command → artifact | `--out` *(req)*, `--hz`(20), `--parse-tokens-re`, `--work-units`, `--work-unit` |
| `report` | provenance + validity + rollup + phase tree | positional `run_dir` |
| `kernel-events` | list GPU/driver-related kernel log lines (read-only) | `--since`, `--until`, `--boot`, `--all-boots`, `--json`, `--max-lines`(50) |
| `measure` | validity-gated utilization vector | `--bytes`, `--flops`(0), `--tokens` |
| `aggregate` | N-run median/IQR/CV% + bootstrap CI (drops INVALID) | `--metric`(kernel_s), … |
| `sample` | adaptive-N until CI half-width &lt; target | `--out-prefix` *(req)*, `--cmd …`, `--until-ci`(3.0) |
| `gate` | nonzero exit unless VALID + CI-tight (+ replicate) | `--min-n`(20), `--max-ci-pct`(5.0) |
| `compare` | 2-artifact hardware diff **or** cross-box REPLICATED | `--replicate-with`, `--metric` |
| `calibrate` | measure BW wall (STREAM-triad), persist per box | `--python`, `--reps`(50) |
| `calibration` | score predicted-vs-measured jsonl | `--pred-key`, `--meas-key` |
| `smoke` | pre-flight bottleneck scan | `--out`, `--marker-re`, `--units`, `--total` |
| `watch` | stall watchdog | `--marker-re`, `--stall-s`(600), `--on-stall` |
| `live` | live GPU/CPU monitor + throttle alarm | `--hz`(2.0), `--duration`, `--once` |

Full flags: `flightrec <cmd> --help`.

---

## Core workflow

### Single run

```bash
flightrec record --out results/run01 -- python bench.py --tokens 128
flightrec measure results/run01 --bytes 105e9 --tokens 128
flightrec report results/run01
```

If `valid=False`, the run throttled (or energy non-monotonic) — **do not average it in**.

### Adaptive-N + gate

```bash
flightrec sample \
    --out-prefix results/my-config \
    --until-ci 3.0 --min-n 5 --max-n 30 \
    --metric kernel_s --tokens 128 \
    --cmd python bench.py --tokens 128

flightrec gate results/my-config_* --metric kernel_s --min-n 5 --max-ci-pct 5.0
flightrec aggregate results/my-config_* --metric kernel_s --tokens 128
```

### Cross-box replication

```bash
flightrec compare results/boxA_run01 results/boxB_run01
flightrec compare results/boxA_* --replicate-with results/boxB_* --metric kernel_s
```

### Monitoring arms

```bash
flightrec smoke --out results/preflight \
    --marker-re 'layer' --units 3 --total 64 \
    -- python long_job.py

flightrec watch --marker-re 'step' --stall-s 600 --on-stall notify \
    -- python long_job.py

flightrec live                 # until Ctrl-C
python -m flightrec.mem_guard --kill-gb 20 --warn-used-gb 90
```

### In-process phases

```python
from flightrec.recorder import FlightRecorder

with FlightRecorder("results/run01") as rec:
    with rec.phase("decode"):
        run_benchmark()
```

---

## Work per joule

`record` can store how much work the command did, and `report` then prints work per joule and joules per work unit:

```bash
flightrec record --out results/run01 --work-units 128 --work-unit tokens -- python bench.py --tokens 128
flightrec report results/run01
```

In-process: `rec.set_work(128, "tokens")` on a `FlightRecorder`. The raw `{units, unit}` goes into `header.json` under `work`; the ratio is computed only in the report, over the same window as the rest of the rollup (the marked phases, else the whole run).

What to keep in mind (from our testing and from reading the code):

- The energy is the **GPU-reported counter** (`energy_mj` from NVML), **not wall power**. It does not include CPU, board or PSU losses, so it should not be compared with wall-socket measurements.
- The counter is sampled at ~6 Hz, so a very short window carries a quantization error at both ends.
- If the run is INVALID (throttled, or the energy counter went backwards), or the counter is missing or did not advance, the report says so and prints no number.
- If no work was recorded, nothing extra is printed.

## Read-only bandwidth

`flightrec calibrate` now also times a read-only sum reduction over a 1 GiB fp32 tensor (about 40x the 25 MB L2) and stores it as `read_only_gbps_measured` beside the existing triad number. The triad keys and file format are unchanged; `header.json` gains `bandwidth_read_only_gbps` (null when not measured). The triad mixes reads and writes, so a read-heavy kernel can exceed it. We therefore treat the triad as a reference for mixed traffic, not as a ceiling for read-heavy kernels. `above_wall` and `pct_of_wall` are still scored against the triad wall only. We have not validated the read-only number across boxes; a failure in this step is recorded as `read_only_error` and does not block the triad calibration.

---

## Kernel events

```bash
flightrec kernel-events --since "2 days ago"
flightrec kernel-events --all-boots --json
```

This only reads the kernel log the systemd journal already keeps (`journalctl -k`, which keeps earlier boots when the journal is set to persistent storage); no service is installed and nothing is enabled. It lists lines as data and does not interpret them. A line is kept if it matches one of these (case-insensitive, first match wins): `nvrm_xid` (`NVRM: Xid`), `nvrm` (other `NVRM:` lines), `thermal` (thermal, overheat), `throttle` (throttl, slowdown), `oom` (out of memory, oom-kill), `watchdog` (hung task, soft lockup, watchdog). `audit:` lines are ignored and everything else is skipped. The output is counts per category (per boot with `--all-boots`, which scans each boot from `journalctl --list-boots` separately and names any boot it could not read), then up to `--max-lines` lines.

`record` also scans the run window when the command finishes. `header.json` gets a `kernel_events` entry (status ok or unreadable with a reason, the count, the window), and any matching lines go to `kernel_events.json`; `report` prints a short section for them. Events do not make a run invalid. If `journalctl` is missing or the journal is not readable for the current user (some systems need membership of the `systemd-journal` or `adm` group), the result is an empty list plus the reason, and the recording is unaffected. Artifacts recorded before this field existed simply print no such section.

What we saw: one line, `NVRM: refcntRequestReference_IMPL: Failed to enter state 1 (current state: 0, status: 0x00000056)`, appeared once each time a `torch.profiler` session started on a GB10. We do not know what it means or whether it matters; listing it is the point of this command, not a conclusion about it.

---

## Artifact layout

Each `--out` directory is self-describing:

| file | contents |
|---|---|
| `header.json` | driver, GPU name, clock ceiling, BW wall source, recorder overhead |
| `samples.parquet` | per-tick GPU/CPU/mem/PSI timeline |
| `phases.parquet` | nested phase marks (if used) |
| `kernel_events.json` | matching kernel log lines for the run window (only if any matched) |

---

## Limitations (GB10 hardware)

- **No live DRAM BW counter** — achieved GB/s is analytic; score vs STREAM wall; watch `above_wall` (L2 residency).
- **NVML throttle bits can fire spuriously** — validity uses **clock + power droop**, not the bitmask alone.
- **NVML memory-info often N/A** on unified LPDDR5X — trust `/proc/meminfo` (`MemAvailable`).
- **Memory clock N/A** via NVML on this platform.
- **No CPU/board power** on stock interfaces.

Deeper measurement protocols (clean-room, on-demand profiling, etc.): [`PROTOCOLS.md`](PROTOCOLS.md).

---

## Layout (this folder)

```text
flightrec/           # Python package
tests/               # pytest (synthetic fixtures; no GPU required)
pyproject.toml
PROTOCOLS.md
systemd/             # example mem_guard unit
examples/baseline/   # optional sample aggregate JSON (illustrative)
LICENSE              # MIT
```

---

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/ -q
```

Suite uses synthetic fixtures — no GPU, no network. Optional plot tests skip without `.[plot]`.

---

## Support matrix

| Platform | Status |
|---|---|
| NVIDIA DGX Spark (GB10, aarch64, unified mem) | **Primary — validated** |
| Other Linux + NVIDIA (e.g. discrete RTX) | Experimental / unvalidated; some constants Spark-shaped |
| Intel GPU / non-NVIDIA | Not supported |

---

*Primary development target: NVIDIA DGX Spark (GB10). Intel / discrete-GPU profiles are deferred.*
