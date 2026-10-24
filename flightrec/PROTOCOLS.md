# Measurement protocols

How to turn the recorder's data into facts. Optimize *with* these, not by eye. "Random or smart ideas
are good, but facts are better" — each protocol is the gate an idea must pass to become a fact.

## 1. A distribution, not a number
- Run **N ≥ 10** iterations; **discard the warmup run** (cold cache + clock ramp).
- Report **median + IQR**, never mean ± std — run times are right-skewed (a throttle or scheduler
  hiccup only ever adds a long tail, never a short one). `flightrec.stats.summarize`.
- **CV% is a diagnostic, not noise to average away.** A high coefficient of variation points at an
  unpinned non-deterministic source (scheduler, CPU boost, throttle). If a change collapses CV, that
  variance *was* a bottleneck.

## 2. A/B = significance AND effect size (both gates)
- "B is 1.5% faster" is meaningless at 3% run-to-run CV. Use the bootstrap CI on the median difference
  (`flightrec.stats.compare`).
- **Two gates, both required:** (a) *statistical* — CI excludes zero (`significant`); (b) *practical* —
  `|rel_pct| >= min_effect_pct` (default 1%, `practical`). At very low CV the CI excludes zero for
  **trivial** sub-1% deltas — seen on the two-box shakedown 2026-06-06, where CPU-boost and THP came
  back "significant" at only ~0.4–0.8% (boost even slightly *inverted*): statistically real, practically
  noise. A genuine lever clears **both** gates.
- An idea that fails either gate is dead, not "probably fine."

## 3. Every run is auto-validated
- The recorder flags a run **INVALID** if the GPU throttled *under load* — either (a) clock < 1400 MHz
  **and** power < 30 W **while** GPU-busy ≥ 50% (the 611 MHz / 13 W PD-wedge signature), **or** (b) clock
  drops more than 50% below the run's own loaded p75 clock (relative droop, hardware-agnostic). Idle
  windows are not throttle and stay VALID. `flightrec.validate.verdict`.
- **Never report tok/s from an INVALID run.** This closes the silent PD-throttle saga.
- The NVML throttle bitmask is deliberately ignored — we saw it fire spuriously on GB10.

## 4. The roofline trajectory
- Achieved bandwidth is **derived** (no GB10 hardware DRAM counter exists): achieved GB/s = bytes moved
  ÷ kernel time. The achievable **wall is MEASURED, not asserted** — `flightrec calibrate` runs a GPU
  STREAM-triad and persists it per-box (box A: **~220 GB/s ≈ 80% of the 273 GB/s spec**, 2026-06-06).
  The old 195 GB/s constant is demoted to the *uncalibrated fallback* only; every artifact's
  `header.json` records `bandwidth_wall_gbps` + `bandwidth_wall_source` so a result states which it used.
- **%-of-wall is the per-run score in the bandwidth-bound regime — stop optimizing when it plateaus near
  the wall.** `roofline.roofline_point()` emits achieved GB/s, %-of-wall, and arithmetic intensity as data.
- **Two axes, not one (2026-06-19).** `bandwidth_regime` is the BANDWIDTH-axis position only (`bandwidth-bound`
  / `mid-wall` / `far-from-wall`) — it is blind to compute and CANNOT tell compute-bound from overhead-bound.
  Supply the **compute ceiling** (`--peak-tflops`, or persist it once with `calibrate --compute`) and
  `roofline_point` adds `pct_of_peak` + the real two-axis **`regime`**: `compute-bound` / `bandwidth-bound` /
  `overhead/latency-bound` / `mixed` (None until the peak is given — it never guesses compute). A 96%-GPU-busy
  run at 5% of peak is `overhead/latency-bound`, NOT compute-bound; only the two-axis regime catches that.
- **Stop hand-modelling bytes/FLOPs.** `flightrec model-bytes <config.json> --m <tok/fwd> --forwards <n> --dtype
  <dt> [--dense-ffn]` derives them (MoE expert traffic via the coupon-collector `E*(1-(1-1/E)^(m*k))`), and
  `measure --from-hf-config <config>` auto-fills them. Validate `active_params_per_token` against the model card.
- **Calibrate the compute peak too.** `calibrate --compute --dtype bf16` measures a matmul peak; quant dtypes
  (`nvfp4`/`fp8`) are recorded with `--peak-tflops <known>` (no torch FP4 matmul — use nvfp4bench's number, e.g.
  GB10 NVFP4 dense ~482 TF). Both persist to the per-box calibration file and `measure` auto-reads them.
- Plot each phase as a point on arithmetic-intensity vs achieved-throughput; colour by verdict so
  bandwidth-starved vs compute-starved vs throttled is visually separable. Label the wall "measured".

### 4b. Cold-cache decode microbenchmarks: ROTATE the working set, report the MEDIAN (what we do)
**GB10 L2 = 25.2 MB** — a single 4096² FP4 weight (~8 MB) sits **entirely in L2**, so a reused-weight loop
measures **L2, not DRAM** (marlin read about 440 GB/s on its own 8.67 MB per call, well above the 273 GB/s memory spec — not possible from DRAM). Real decode reads
each layer's weight **cold from DRAM** (the model is GBs; long-evicted by the time a token returns). The
protocol we settled on (2026-06-06; we compared three methods and rotation looked the most trustworthy to us, see the caveat on the numbers below):
- **ROTATE over a weight POOL larger than L2** (≥ ~64 MB, e.g. 8 weights): each timed call uses a fresh
  weight that's been evicted since its last use. Looked clean in our runs (CV ~2%, though the raw samples were not saved), with no obvious perturbation. **This is our best current method, not a proven one.**
- **Report the MEDIAN + bootstrap CI** (`stats.median_ci`), **never best-of** (best-of games the L2 residual).
- **We use rotation rather than a big-buffer "flush" between calls** (e.g. a 256 MB write). Marlin with rotation read
  **50 µs (reported as 85% of a 220 GB/s wall; see the review update on the byte count below; one run, median of 48 samples, gx10)** in the original June log. **Re-run 2026-10-23** with raw
  samples saved (marlin, M=1, 4096², n = 200 per cell, two sessions on gx10 and one on gx11): rotation over 4 to 32 weights read
  about **46 to 50 µs** and was flat across pool sizes; a pool of 2 weights (under the L2) read about 19.8 µs, the same as a
  single weight in a loop; the 256 MB flush read about **87 to 90 µs**, about 1.8 times the rotation figure. The earlier note
  that flush was *noisier* was **not** reproduced (its spread was about as tight as rotation's). Why flush reads slower is
  untested; one guess is that the 256 MB write leaves dirty lines being written back during the timed call.
- **Review update (2026-10-23): the timings above include launch overhead.** They put CUDA events around one Python call, so each
  number contains the cost of launching it. Re-timing with CUDA-graph replay (M=1, rotation over 8 weights, n = 200, one session
  per box) gave, in µs on gx10 / gx11: marlin 37.8 / 39.7, mm_fp4 53.1 / 51.3, a naive single-buffered custom kernel
  125.8 / 103.8. The launch overhead was about 9 µs for marlin and about 45 µs for mm_fp4, so event timing overstated mm_fp4 the
  most. The event-timed figures above are kept for the record, but their ratios (mm_fp4 about 2 times slower than marlin) should
  not be used; with graph timing mm_fp4 looked about 1.2 to 1.7 times slower than marlin at M=1 (the range over launches, boxes and shapes, see below). Part of the early difference was our own harness, which
  recomputed mm_fp4's scale factor inside every timed call (two extra small kernels). With that moved out, a re-run on gx10
  gave mm_fp4 48.2 µs against marlin 39.2 µs (a ratio of 1.23), while later launches of the same script gave 1.33 to 1.45, and a profiler showed the GEMM kernel alone at 47.0 µs against
  marlin's 39.7 µs. We found the launch-overhead problem by accident, because
  a second timing loop disagreed with the first. We now suggest checking any timing with a second, independent method (graph replay or a
  profiler's kernel duration) and treating a disagreement as a finding.
- **Review update: timings vary between process launches.** The 200-sample intervals above describe one launch. Repeating the same script in a fresh process (4096x4096, M=1, several launches per box over about 13 hours, including one power cycle) changed the per-call median by 4 to 8% for marlin, 1 to 9% for mm_fp4, 16 to 24% for our single-buffered custom kernel and 37 to 39% for the double-buffered one (up to 43% in some cells). The ratio of mm_fp4 to marlin at M=1 was 1.23 to 1.45 on gx10 and 1.29 to 1.39 on gx11 over these launches, and 1.54 to 1.69 for the 4096x12288 and 12288x4096 shapes in the final pass. In a follow-up test on gx10, reallocating the weight pools 12 times inside one process (a dummy allocation of random size kept alive between rounds to shift later allocations) changed the M=1 per-call median by 22 to 37% for mm_fp4 and our two custom kernels, but by 0.9% for marlin. The spread therefore appears to depend on where the weights are allocated; we have not observed physical addresses or identified the mechanism. For cross-kernel comparisons we now suggest repeating the allocation several times and reporting the range, not a single launch. The lowest-time kernel at each batch size was the same in the sessions compared.
- **Review update: flashinfer's autotune changes the `mm_fp4` configuration at small M.** All `mm_fp4` figures above come from the default call. In every launch we profiled, the default call runs one CUTLASS configuration (tile 128x128x128, 4 pipeline stages). With `flashinfer.autotune(True)` (flashinfer 0.6.8.post1, gx10), the call at M up to 32 switches to a 2-stage variant of the same tile. In fresh processes, with and without autotune and 3 fresh allocations each, the per-call median at 4096x4096 went from 46.7 / 47.6 µs to 42.6 / 44.3 µs at M=1, from 45.7 / 46.0 to 43.2 / 43.2 µs at M=16 and from 48.2 / 49.2 to 44.0 / 43.2 µs at M=32; there was no change at M=64 and M=128. Marlin in the same harness read 37.8, 38.3 and 40.7 µs at M=1, 16 and 32, so the M=1 ratio of `mm_fp4` to marlin is about 1.13 to 1.17 when tuned, against about 1.2 to 1.45 for the default call. The autotune cache persists inside a process, so later default calls in the same process are already tuned. One machine and one shape.
- **Review update: the byte count and the bandwidth "wall".** The "% of wall" figures used one byte count for every kernel, 9.45 MB per call at M=1: 4-bit weights plus one 8-bit scale per 16 values, which is the NVFP4 count. Marlin moves 4-bit weights plus 16-bit scales per 128 values, about 8.67 MB, so its percentages were about 9% too high. On its own count, the graph-timed 37.8 µs for marlin is
  about 229 GB/s, above the 220 GB/s constant used for "% of wall" here and elsewhere in these notes but below values measured on other days on gx10 (STREAM-triad 235 GB/s; a read-only sum reduction, best of 50, fp32, 247 to 263 GB/s). So the 220 GB/s constant is not a ceiling, and
  percentages computed against it (including the 85% above) should not be used. Against 263 GB/s, marlin at 37.8 µs is about 87%, and mm_fp4 on its own 9.45 MB at about 43 to 47 µs is about 76 to 83%.
  The 263 GB/s figure is itself one session's best of 50 on fp32, so it is an upper envelope and not a guaranteed ceiling.
- **Consequence:** the cold ranking can differ wildly from hot AND from a bad cold method, and from a timing method that includes
  launch cost. In a batch-size sweep (M = 1 to 512, 4096², graph-timed, one session per box) marlin had the lowest time per
  token up to M = 64 and mm_fp4 from M = 96 up, on both boxes. Further sessions and two other shapes (4096x12288 and
  12288x4096) showed the same pattern, apart from the 12288x4096 shape at M = 128, where marlin was ahead on both boxes. Our two naive custom kernels were not ahead in any band we
  could reproduce, so we make no claim that they beat marlin or mm_fp4 anywhere. These are single sessions; the spread
  between boxes and sessions is larger than the within-run intervals. We guess (untested) that the custom kernels are
  occupancy-bound at small M. Any cross-kernel head-to-head should use identical rotation+median on both sides (we think),
  or it is not a comparison. (This protocol itself took 5 corrections to get right — hot→best-of→flush→rotation→graph timing.)

## 5. Find, don't guess — the escalation ladder
When a window is slow, **descend until you hit a layer you can change, or have good evidence is silicon:**
```
app / config  →  CUDA kernel  →  driver / runtime  →  OS / kernel  →  firmware  →  silicon
                                                       (governor,       (PD /        (273 GB/s wall,
                                                        isolcpus,        PROCHOT)     99 KB SMEM,
                                                        hugepages,                    no tcgen05)
                                                        IOMMU)
```
Each step is a measurement, not an opinion. Stopping at the first plausible cause is how you "optimize"
a kernel that was actually starved three layers down. The recorder is the sensor at every rung.

## 6. Two measurement levels (by physics, not preference)
- **Always-on telemetry (this tool; earlier called Tier-1; no privilege):** broad cheap state, every run, about 3.4% of one core (measured; see the overhead section below).
  Records "every second" — but *not* "every cycle which cores": that needs profiling-grade tooling.
- **On-demand profiling (earlier called Tier-2; sudo):** `nsys --gpu-metrics-set=gb20y` for SM/Tensor-active time-series; `ncu`
  for per-kernel roofline. Perturbing (counter multiplexing, kernel replay) — zoom in only where
  the telemetry flags a window. `nsys` `systemClockNs` == `CNTVCT_EL0`, which leads `CLOCK_MONOTONIC_RAW` by a
  fixed ~+19.32 s; rebase by that offset to fuse profiler traces onto the telemetry timeline.

## 7. Exploit both Sparks (the dual-box bonus)
Two matched GB10s linked at ~200 Gb/s is a force-multiplier on rigor and throughput. Fabric setup:
see sibling [`../dual-spark/`](../dual-spark/). Three modes, in increasing value (record on both boxes,
collect artifacts, then `flightrec compare`):
1. **2× throughput** — partition the config grid across boxes; halve a restart-bound sweep's wall-clock.
2. **Replication = rigor** — run each survivor on *both*; promote to "fact" only if the win replicates
   within overlapping CI on both (`compare_distributions`). Kills box-specific flukes.
3. **Control vs treatment** — hold one box untouched as baseline, perturb the other (OS/kernel/firmware);
   `compare` isolates the change's effect from ambient drift.

**Prerequisite — calibrate first (experiment #0):** you cannot use one box as a control/replicate until
they look equivalent. Run an idle twin compare; the **provenance diff must show no `DIFFERS`**
(checked 2026-06-06: both driver 580.159.03, ceiling 3003 MHz — twins matched). And BOTH boxes must
start from the **same workload state** — a box carrying a resident model (e.g. vLLM) is not a clean
control until it is cleared.

## 8. One GPU memory consumer at a time (GB10 unified-memory pre-flight)
GB10's 128 GB is **unified** (~120 GB usable, CPU+GPU share one pool), so a campaign and the model it
measures draw from the same memory. Two traps, both seen live (2026-06-08, unified-memory footprint investigation):
- **`--gpu-memory-utilization` is a fill target, not a fit limit.** vLLM grows its KV-cache pool to consume
  that fraction of the *whole box* — at `0.8` an omni server whose weights are only 20.9 GB expands to
  **~96–102 GB**, parking the box at the wall. **Cap it: `--kv-cache-memory-bytes` (≈17 GiB) → ~45 GB
  footprint.** Standing rule: **any single model footprint <100 GB**; cap KV bytes, never trust util alone.
- **The measurement stack is itself a GPU consumer.** Embedder + reranker + their transient batch spikes
  are several GB; on top of a near-full box that is enough to hit `NV_ERR_NO_MEMORY` → kernel `SLUB …
  GFP_ATOMIC` → **systemd-oomd reaps ~25 service cgroups** (incl. `nvidia-persistenced`, `autofs` — the
  latter silently kills network/autofs fabric mounts). It respawns them: **looks like a "reset" but the box never
  rebooted.** So: **never co-reside two consumers** — serialize (serve → stop → measure), or cap util ~0.4
  and bound batch size. **Pre-flight every launch** with `nvidia-smi --query-gpu=memory.used` first.

**Start-safe practice:** clamp fractional `gpu_memory_utilization` to a headroom ceiling (e.g. ≤0.80)
unless the box is solo and the recipe has shown it needs a larger pool. Prefer explicit
`--kv-cache-memory-bytes` for co-resident helpers so a second consumer cannot fill the unified pool.
Some community launchers implement this as an env-tunable util ceiling; the principle matters more than
the wrapper.

## 9. Pre-flight before you commit (`flightrec smoke`)
- **Never commit a multi-hour job blind.** Before a long quant / training / bench run, scan a SHORT
  representative slice with `flightrec smoke --out <dir> --marker-re '<per-unit line>' --units K --total N
  -- <cmd>`. It records the first K marker units under the recorder, kills the job (SIGTERM→SIGKILL grace),
  and emits a one-screen card: **verdict + knobs + full-run ETA**. `flightrec.smoke`.
- **The verdict ladder** (first match wins, `smoke.diagnose`): **THROTTLED** (clock/power collapsed →
  measurement invalid, cold-drain & retry) → **INCONCLUSIVE** (GPU loaded <50% of the slice → marker fires
  before real work; fix `--marker-re`) → **GPU-STARVED** (<25 W under load → data-movement/CPU-bound, e.g.
  `low_gpu_mem_usage=True`) → **NEAR-OOM** (<8 GB unified free → OOM-reset risk, lower batch; a CAPACITY signal,
  not bandwidth) → **GPU-BUSY** (≥80% duty → work IS reaching the GPU) → **INTERMEDIATE**. This is the one place
  flightrec draws a conclusion; it is explicitly heuristic and prints the raw signals alongside.
- **`GPU-BUSY` is NOT `compute-bound` (2026-06-19 rename).** Duty-cycle reads ~100% for a launch/latency-bound
  job too (many tiny kernels) — the old `COMPUTE-BOUND` label over-claimed. To actually confirm compute-bound,
  compare achieved GFLOP/s to the measured peak (the two-axis `regime`, §4), not duty alone. Likewise the old
  `MEMORY-BOUND` is now `NEAR-OOM` (it was about free-byte headroom, never bandwidth).
- **ETA is steady-state, not first-unit.** Per-unit time is the median of marker intervals with the first
  `--settle` (default 1) dropped — first-unit JIT/allocator cost never enters the extrapolation.
- **Concrete payoff:** a Qwen3.6-35B AutoRound quant ran ~7h at 15 W (`low_gpu_mem_usage=True` starved the
  GPU). A 5-min smoke flags GPU-STARVED → `low_gpu_mem_usage=False` before the commit. Pairs with predict→prune (§4).
- **Thresholds are tunable module constants** (`STARVE_POWER_W`, `MEM_HEADROOM_FLOOR_MB`, `COMPUTE_BUSY_PCT`,
  `UPSIZE_HEADROOM_MB`); revisit them against the first real quant/training slices, don't treat as gospel.

## 10. Watch a long run for stalls (`flightrec watch`)
- **Long runs go through flightrec, never raw `nohup`.** A multi-hour quant/train can HANG — process
  alive but GPU at 0 W / 0 %, no progress for hours — and nothing flags it. `flightrec watch` is the
  during-the-run alerting layer over the same GPU timeline (`flightrec.watch`).
- **STALLED is a conjunction, by design:** flagged only when BOTH (1) the progress marker (`--marker-re`,
  same as smoke's) has gone silent for `--stall-s` AND (2) the GPU has sat continuously idle
  (`power < --idle-power-w` and `duty < --idle-busy-pct`) for `--stall-s`. Requiring both separates a true
  hang from a legitimately-long single kernel (which keeps the GPU BUSY → no false alarm). Bias is
  conservative: a hung job that still draws power is MISSED rather than a working job being false-killed.
- **Two source modes:** **wrap** — `flightrec watch --marker-re '<line>' -- <cmd>` (run + watch the child);
  **attach** — `flightrec watch --marker-re '<line>' --logfile <path> [--pid <pid>]` (tail an
  already-running raw-nohup job, liveness via `/proc/<pid>`).
- **On stall:** notify (always, to stderr + `--log`); optional `--checkpoint-cmd` / `--notify-cmd` shell
  hooks (`FLIGHTREC_STALL_MSG` in env); `--on-stall kill` does SIGTERM→SIGKILL (`--grace`). `--record <dir>`
  writes the poll timeline (flat power + frozen mark count) as a post-hoc record.
- **Concrete payoff:** an auto_round re-quant hung at block 16 (inductor/torch_compile stall), discovered
  ~2 h later via a manual check because it was a raw `nohup`. A 10-min `--stall-s` would have flagged it
  12× sooner. Complements the arms: `quiesce` = before measure, `smoke` = before commit, `memguard` = OOM
  backstop, `watch` = during run.
- **Defaults are tunable** (`STALL_TIMEOUT_S=600`, `POLL_INTERVAL_S=5`, `WARMUP_S=120`, `IDLE_POWER_W=20`,
  `IDLE_BUSY_PCT=10`); warm-up grace covers idle weight-load at startup — raise it for very large models.

## 10b. Is the box at rest? (`flightrec quiesce`)
- **A measurement on a busy box is skewed even when nothing throttled** — a co-resident embed rebuild or a
  second model shares the GPU/CPU and inflates kernel time. `validate` catches THROTTLE; `quiesce` catches
  CONTENTION (the dual failure). Every other vital is uninterpretable without it.
- **Gate before you measure:** `flightrec quiesce && flightrec record …`. It samples a short baseline of
  foreign GPU duty/power + 1-min loadavg per core, returns **QUIET / CONTENDED** + reasons, and **exits
  nonzero when contended** so it composes in a shell. GB10 has no per-PID GPU accounting (NVML compute-apps
  is N/A), so it reads the box's resting vitals rather than subtracting the foreigner.
- **Floors are tunable** (`--busy-floor 10%`, `--power-floor 25 W`, `--load-floor 0.5/core`; mirror the
  `watch` idle thresholds). Seen 2026-06-19: it flagged a contended box and passed a quiet one.

## 11. Kernel tree via nsys, on-demand profiling (correlate, don't re-instrument)
- **The recorder already emits NVTX** (`rec.phase()` → `torch.cuda.nvtx.range_push/pop`), so the telemetry
  envelope and the profiler's per-kernel tree are the SAME phases — no new code, just the invocation:
  ```
  nsys profile -t cuda,nvtx -o results/run01/trace \
      flightrec record --out results/run01 -- <bench ...>
  ```
  Open `trace.nsys-rep` in the Nsight Systems GUI: every CUDA kernel, duration, and warp occupancy shows
  up *inside* the `decode`/`prefill` NVTX ranges you marked. The telemetry gives the power/clock envelope; nsys
  gives the kernel breakdown within it — the "program logic tree" path.
- **Fuse the two timelines** with the §6 offset: nsys `systemClockNs` (== `CNTVCT_EL0`) leads
  `CLOCK_MONOTONIC_RAW` by a fixed ~+19.32 s on this box; rebase by that to overlay a telemetry lane spike
  onto the exact kernel that caused it. Zoom in only where the telemetry flagged a window — nsys perturbs.

## Overhead is a measured, controlled constant
The telemetry's cost is uniform across runs, so it cancels in any A/B comparison (you only ever compare
relative numbers). It is **measured, never claimed**: every run writes its own thread-CPU cost to
`header.json` as `recorder.overhead_pct_of_core` (plus tick count, mean period, overrun %). On a reference GB10 it
is **~3.4% of one Grace core** — *not* the "<1%" once asserted (that was ~8× off). The dominant cost was
`nvmlDeviceGetTotalEnergyConsumption` at ~1.6 ms/call; because energy is a monotonic integral used only
as a max-min delta, it moved to the ~6 Hz slow path for **zero information loss** and a 59% overhead cut
(8.1% → 3.4%, reference GB10 2026-06-06). Leave it on for every run; turn it off for the final clean-room number
and reclaim that fraction. Profiling is *not* uniform — that is why it stays on-demand.
