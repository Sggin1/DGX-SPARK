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
| `kernel-events` | list GPU/driver-related kernel log lines (read-only) | `--since`, `--until`, `--boot`, `--all-boots`, `--json`, `--max-lines`(50), `--guard`, `--group`, `--max-groups`(20), `--log [PATH]`, `--since-last` |
| `events-log show` | read the rolling event log (live + rotated) | optional `PATH`, `--last`(20), `--group` |
| `platform` | print a read-only platform snapshot | `--json` |
| `platform-diff` | show only the fields that differ between two snapshots | positional `A B` |
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

## Counted groups and an optional event log

```bash
flightrec kernel-events --since "2 days ago" --group              # counted stack, top 20 groups
flightrec kernel-events --all-boots --group --guard --max-groups 10
flightrec kernel-events --log --since-last                        # append to the default log file
flightrec events-log show --last 20
flightrec events-log show --group
```

On one GB10 the kernel log held 173 lines of the form `NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:NNN`, often several in the same second and differing only in numbers. Listing each one is noise, so `--group` (works with `--all-boots`, `--json`, `--guard`) prints one line per normalized message:

```
173 x [nvrm_oom] NVRM: nvCheckOkFailedNoLog: ...  (first <time>, last <time>, largest burst N/s)
```

The key replaces timestamps, PCI addresses, `0x...` numbers, decimals, pids in brackets and digit runs of 3 or more with placeholders (`<ts>`, `<pci>`, `<hex>`, `<n>`, `[<pid>]`) and collapses whitespace; messages with the same key and category are counted together, and the example is the earliest raw line. Shorter numbers are kept, so `Xid 79` and `Xid 13` stay separate groups; memory-guard lines are normalized more aggressively because they are mostly numbers. "Largest burst" is the most occurrences inside one calendar second as the journal timestamps them (second resolution). Groups are sorted by count, then first time, and `--max-groups` (20) caps the printout. With `--group --json` the output carries `groups` instead of `events`.

`report` now shows the top few groups with counts in the kernel-events and memory-guard sections instead of the first raw lines. `kernel_events.json` / `memguard_events.json` still hold raw lines, but at most 2000 each; if more were found, `header.json` records `truncated: true`, `stored: 2000` and the true `count`, and `report` says that the group counts cover only the stored lines.

### Rolling event log (off unless you pass `--log`)

```bash
flightrec kernel-events --log [PATH] [--max-bytes N] [--backups N] [--no-compress] \
                        [--max-age-days N] [--since-last] [--quiet]
```

`--log` appends one JSON line per group per scan: `{ts, boot, category, key, example, count, first, last}` (`host_label` is a supported field of the module but is not set by the command line). Without PATH the file is `$XDG_STATE_HOME/flightrec/events.jsonl`, or `~/.local/state/flightrec/events.jsonl` when that variable is unset or not absolute; nothing is written under `/tmp`. With `--guard` the guard groups are logged too. Limits as implemented:

- Merge: if the file's final line has the same key and boot and its `last` is within 60 s of the new group's `first`, the counts are added and `last` updated in that line instead of adding one. Only the final line is considered, so with several different messages per scan this merges less often than one might hope.
- Size: when an append would pass `--max-bytes` (5 MiB) the file is renamed to `events.jsonl.1.gz` (`.1` with `--no-compress`), older backups shift up, and anything beyond `--backups` (5) is deleted. A single record larger than the cap is still written.
- Age: `--max-age-days N` deletes rotated files whose modification time is older than N days (the live file is never pruned). Default is no age limit.
- Locking: an advisory `flock` on `events.jsonl.lock` where Python has `fcntl`; otherwise no lock is taken. We have only exercised the lock path with a stand-in object in tests, not with concurrent writers.
- Errors: a failed write prints `event log: could not write (...)` and exits 1; it does not raise and does not affect the scan.
- `boot` is the journalctl offset at scan time (0 = current), so the same boot is numbered differently after a reboot.

`--since-last` (needs `--log`) stores the newest event time seen in `events.jsonl.state` and starts the next scan one second after it, so repeated runs mostly add only new events. It uses event times, not a journal cursor. Events written to the journal within that same second after a scan could be missed, and a run that reads without `--since-last` over a range already logged will count those events again. If a boot cannot be read, the stored time is not advanced. When a stored time exists it replaces `--since`.

`events-log show [PATH] [--last N] [--group]` reads the live file and any rotated `.gz` or plain backups, oldest first. Without `--group` it prints the last N records (20); with `--group` it sums all records (or the last N if given) into counted groups. Burst figures are not stored in the log, so they are not shown there. Unreadable lines are skipped and counted.

Memory-guard's own `--log` file (lines like `[HH:MM:SS] WARN used=...`) is not read: those lines carry no date, and labelling them with the file's modification date would be a guess. `mem_guard.py` is unchanged.

Nothing is scheduled or installed by flightrec. If you want periodic scans, the following is only an example to adapt; we have not run it on a real box.

```
# EXAMPLE only - cron, every 10 minutes, current user
*/10 * * * * flightrec kernel-events --all-boots --guard --log --since-last --quiet

# EXAMPLE only - ~/.config/systemd/user/flightrec-events.service
[Service]
Type=oneshot
ExecStart=flightrec kernel-events --all-boots --guard --log --since-last --quiet

# EXAMPLE only - ~/.config/systemd/user/flightrec-events.timer
[Timer]
OnCalendar=*:0/10
Persistent=true
[Install]
WantedBy=timers.target
```

The journal must be readable by that user (see above). User-level units may need `loginctl enable-linger` to run when logged out.

---

## Platform snapshot

```bash
flightrec platform --json > before.json     # after a firmware/OS update: > after.json
flightrec platform-diff before.json after.json
```

Firmware or OS updates may change usable memory or add monitoring features, and we would like each run to record enough to see that afterwards. `record` stores a `platform` entry in `header.json`; `flightrec platform` prints the same thing for the current machine. It is raw values, each read on its own (an unreadable field is `null` and the others are unaffected), with no sudo: kernel release (`/proc/sys/kernel/osrelease`), BIOS version, date and product name (`/sys/class/dmi/id/`), `MemTotal`, `MemAvailable`, `CmaTotal` and `HugePages_Total` (`/proc/meminfo`), the kernel's boot line `Memory: <avail>K/<total>K available (... reserved, ... cma-reserved)` from `journalctl -k -b 0` (raw line plus the parsed numbers), the first kernel-log line containing `mpam:`, whether `/sys/fs/resctrl` exists, the sorted names under `/sys/bus/event_source/devices`, and the NVIDIA kernel module version from `/proc/driver/nvidia/version`. Hostnames, usernames, serial numbers and UUIDs are not read. If the journal is not readable, the boot-log fields are `null` and `kernel_log_error` says why.

`report` prints a short `platform:` block (kernel, BIOS, MemTotal, boot reserved kB, the MPAM line or "none seen", resctrl yes/no, number of perf devices). `platform-diff A B` takes two `header.json` files or two `flightrec platform --json` files and prints only the fields whose values differ, or `no differences`. It reports that values differ, not why. Artifacts recorded before this field existed print no platform block, and `platform-diff` says they have no snapshot.

---

## Memory guard in the run report

```bash
flightrec kernel-events --since "2 days ago" --guard
```

`flightrec-memguard` (`flightrec.mem_guard`) can terminate processes. So that a cutoff is visible next to the run it happened during, `record` reads `journalctl -u flightrec-memguard` for the run window when the command finishes. `report` prints `memory guard during the run: none`, `N events`, or `could not read (reason)`, followed by the first few lines; `header.json` gets a `memguard` entry (status, reason, count, window) and any lines go to `memguard_events.json`. `kernel-events --guard` lists the same lines for the `--since`/`--until`/`--boot` range you give it.

Only these shapes, as `mem_guard.py` prints them, are kept: `WARN used=...`, `BREACH avail=...`, `SIGKILL <pid>: <command line>` and `docker stop <id> (<name>)`. Start-up (`guard up: ...`) and systemd start/stop lines are ignored. Guard events are only listed: they do not make a run invalid, and nothing here changes, signals or configures the guard. The `SIGKILL` line carries the killed command line as the guard printed it, so it can contain paths or arguments; check it before sharing an artifact.

What to keep in mind: `mem_guard.py` writes its messages to the file given by `--log` (default `/tmp/flightrec_memguard.log`), and the example unit in `systemd/` does not pass `--log`. We have not verified that those lines reach the journal on any particular install, so "none" here can mean "nothing was logged to the journal" and not "the guard did nothing". The unit would need to send its output to the journal for this section to see events.

---

## Artifact layout

Each `--out` directory is self-describing:

| file | contents |
|---|---|
| `header.json` | driver, GPU name, clock ceiling, BW wall source, recorder overhead, `platform` snapshot, `memguard` status |
| `samples.parquet` | per-tick GPU/CPU/mem/PSI timeline |
| `phases.parquet` | nested phase marks (if used) |
| `kernel_events.json` | matching kernel log lines for the run window (only if any matched) |
| `memguard_events.json` | memory-guard warn/kill lines for the run window (only if any matched) |

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
