# DGX-SPARK

DGX Spark research and tests — containers, benchmarks, and investigation notes for running large models on the NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory).

Entries address compatibility issues with CUDA 13.x, aarch64, and SM121 that aren't covered by upstream containers or documentation. Each folder is a self-contained topic; dates and environment details live inside each folder's README.

**Scope of the results.** Results in this repository come from the author's own measurements on their own hardware, on the dates given. Terms such as "verified" or "confirmed" mean the result was reproduced in the stated runs (often once or a few times); they do not mean independently confirmed or general. Sample sizes are small and some raw data was not kept, so treat the figures as leads to check on your own setup.

## Contents

The GB10 software stack changed quickly across these months: NVFP4 and CUDA support moved from community workarounds to upstream support. The results in this repository reflect the software state up to about 2026-06-22. Each result carries the dates and versions given with it, and should be rechecked against current software before it is relied on. Where a folder does not record when its tests ran, the date shown is from the repository history and is marked as such.

At a glance (details, dates and scope are in each folder):

| Folder | Topic | Type of content |
| --- | --- | --- |
| [`nvfp4/`](nvfp4/) | NVFP4 on GB10 | notes and measurements |
| [`turboquant/`](turboquant/) | 3-bit KV cache compression test | measurements (single runs) |
| [`mamba-dev/`](mamba-dev/) | mamba-ssm build for aarch64 | container build |
| [`nemo3-super-gguf/`](nemo3-super-gguf/) | Nemotron-3-Super via llama.cpp on sm_121 | build recipe and measurements |
| [`dual-spark/`](dual-spark/) | pairing two GX10 over one CX-7 cable | runbook |
| [`atlas/`](atlas/) | Atlas engine vs vLLM | benchmark (one host) |
| [`fp4_SASS/`](fp4_SASS/) | which FP4 tensor-core paths work on sm_121a | compiler and SASS evidence |
| [`comfyui_spark_notes/`](comfyui_spark_notes/) | ComfyUI wheel-shadowing | observations on one install |
| [`recipes/`](recipes/) | serving launch files | links to community recipes, local measurements |
| [`flightrec/`](flightrec/) | run validity, memory guard, monitoring | tool |

### 1. [nvfp4/](nvfp4/) — NVFP4 on DGX Spark (friction → working)

**Tested:** lab work 2026-03-25/26; accuracy banners 2026-06-05; narrative updated 2026-08-06. Driver 580.142 then 580.159.03; CUDA 13.0 to 13.2. Support was changing quickly during this period; the folder records what held at each date.

Single top-level home for the NVFP4 story on GB10 (only `nv*` topic folder). March 2026 memory-floor campaign (19 GB model, 50–120 GB → ~32 GB @ ~50 tok/s), then the **software + driver/OS catch-up** that made stock upstream viable. Former trees `nvfp4-guide/`, `nvfp4-landscape/`, `nvfp4-memory/` live under this folder (`guide.md`, `memory.md`, `archive/`).

- Current narrative + what still holds: [`nvfp4/README.md`](nvfp4/README.md)
- Living launch YAMLs: [`recipes/`](recipes/) (not re-copied here)
- Silicon deep dive: [`fp4_SASS/`](fp4_SASS/) (related FP4 silicon notes; separate topic)
- Dated lab notes under [`nvfp4/archive/`](nvfp4/archive/) (March 2026 stamps)

### 2. [turboquant/](turboquant/) — TurboQuant 3-bit KV Cache Compression

**Tested:** March 2026 (point-in-time snapshot, as stated in the folder). Driver 580.142, CUDA 13.2, the vLLM state and PR patch of that time; not re-run on current builds.

Patches vLLM with TurboQuant KV cache compression ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)). Builds on the NVFP4 stack — same model family, compressed KV cache.

- 240K-token context fits at 64 GB; no memory creep seen over 8 sequential queries in one server session (not a long-running soak test)
- 7.4 vs 5.0 tok/s (TQ3 vs FP8) at 64K context; each context length measured once, so small differences are within noise
- Needle-in-haystack recall tested across 1K–240K tokens (the 240K run is a capacity result at 1.8 tok/s, not a speed result)
- Naming: the tested patch adds `tq3` and `tq4` cache dtypes; the title of the merged upstream PR describes 2-bit compression, so the variant in the merged version may differ from the one tested here
- Based on [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with prebuilt SM121 FlashInfer

### 3. [mamba-dev/](mamba-dev/) — mamba-ssm for aarch64

**Dated:** test dates are not recorded in the folder; first committed 2026-03-30, last updated 2026-08-06 (repository history). CUDA 13.0 to 13.2; describes the PyPI mamba-ssm 2.3.1 behaviour of that time.

Working mamba-ssm + causal-conv1d build for DGX Spark. `pip install mamba-ssm` from PyPI is broken on aarch64 (x86_64 binaries + PyTorch ABI mismatch). This container builds from source.

- Loads NemotronH hybrid models (Mamba-2 + Attention) via transformers
- Tested with Nemotron-Nano-12B-v2-VL-BF16 (13.2B params on GPU)

### 4. [nemo3-super-gguf/](nemo3-super-gguf/) — Nemotron-3-Super 120B via sm_121 llama.cpp

**Tested:** benchmarks dated 2026-03-14; package committed 2026-04-12. Driver 580.126.09, CUDA 13.0 as stamped in the folder. The NVFP4 blockers described were the state of vLLM at that date.

Native sm_121 llama.cpp build for Nemotron-3-Super 120B MoE. GGUF path documented because the NVFP4 checkpoint was blocked on vLLM — NemotronH LatentMoE uses relu² with separate projections and no fused `act_and_mul` MoE backend supports it.

- ~17 tok/s at Q4_K (66 GB weights, ~71 GB runtime)
- Ollama GGUF not compatible with upstream llama.cpp (different MoE tensor layout)
- NVFP4 attempt documented: three layers of failure (config, kernel, pip wheels)

### 5. [dual-spark/](dual-spark/) — Pairing two GX10 over a single CX-7 cable (200 Gb/s RDMA)

**Dated:** test dates are not recorded in the folder; first committed 2026-05-07, last updated 2026-08-06 (repository history). Two ASUS Ascent GX10 on DGX OS of that period.

Field-tested runbook for pairing two NVIDIA DGX Sparks (ASUS Ascent GX10) over a single CX-7 direct-link cable. Covers the two failure classes that burn the most forum hours: power-path throttling (measure with `gpu_stress.py` — AC sag can clamp P8 under load; not all UPS bad) and dual-node connection issues (UFW, NCCL TCP-fallback, GID-index, half-bandwidth twin misconfig). Username models A (same user), B (two users + SSH map), and B→A conversion are documented.

- Symptom → fix table at the top of [`dual-spark/DUAL_SPARK_SETUP.md`](dual-spark/DUAL_SPARK_SETUP.md)
- Visual cable / twin / PCI map: [`dual-spark/TOPOLOGY.md`](dual-spark/TOPOLOGY.md) (one cable ≈ 200 Gb/s total)
- Measured: ~195–197 Gb/s RDMA aggregate (about a full single-cable budget)
- Patches eugr's launcher with NCCL multi-rail + RoCEv2 GID overrides

### 6. [atlas/](atlas/) — Atlas Inference Engine: head-to-head vs vLLM

**Tested:** 2026-05-07. Driver 580.142, CUDA 13.0; engine and image versions as listed in the folder. Historical snapshot: upstream Atlas has had releases since (for example b470 on 2026-09-24), so these figures may no longer apply; current status is on the [Atlas releases page](https://github.com/Avarok-Cybersecurity/atlas/releases).

Single-host benchmark of [Atlas](https://github.com/Avarok-Cybersecurity/atlas), a pure Rust+CUDA inference engine for GB10. Measured 2026-05-07 on one ASUS Ascent GX10, same model and prompts for both engines. The `~88 tok/s` figure published for Nemotron-3-Nano-30B-A3B-NVFP4 was reproduced (88.34 max, 84.36 median, c=1). Single-stream (c=1, 9 runs each): Atlas median +62.7% over vLLM. At c=4 (9 runs each): vLLM aggregate +53% over Atlas, consistent with continuous batching. Cold start with cached weights, as measured: 20 s (Atlas) vs 171 s (vLLM). Includes reproducer (`bench.py`), raw JSON results, and the `serve --help` flag dump.

### 7. [fp4_SASS/](fp4_SASS/) — FP4 on GB10: which tensor-core paths work

**Dated:** test date not recorded in the folder; first committed 2026-05-18 (repository history). Driver 580.142, CUDA 13.2 toolchain, CUTLASS DSL 4.5.1 as stamped in the findings.

Evidence-first probe of what FP4 tensor-core paths actually work on `sm_121a`. In our testing, NVIDIA's own `ptxas` refuses `tcgen05.mma` for `sm_121a`, which looks like a silicon-level gap on GB10 rather than a software setting (evidence trail in `findings.md`). Consumer warp-level FP4 (`mma.sync.aligned.kind::mxf4.block_scale...`) compiles cleanly and emits the same SASS opcode (`OMMA.SF.16864.F32.E2M1.E2M1.E8`) as the RTX 5060 Ti (`sm_120a`) — Spark and the 5060 Ti share the consumer-Blackwell FP4 silicon at the instruction level.

- `findings.md` — full evidence trail with PTX/SASS citations
- `kernel-patterns.md` — patterns for integrating custom low-bit formats with consumer-Blackwell FP4, distilled from llama.cpp MMQ
- `reference/fp4_mma_reference.cu` — working MXFP4/NVFP4 inline-PTX MMA, compiles cleanly on sm_120a + sm_121a
- CuTeDSL 4.5.1+ now supports `sm_121a` as a first-class JIT target (earlier 4.4.x "missing kernel images" reports no longer apply)

### 8. [comfyui_spark_notes/](comfyui_spark_notes/) — ComfyUI wheel-shadowing gotchas on Spark

**Observed:** 2026-05-22 on one ComfyUI install.

Two real pip-shadowing risks observed while running ComfyUI on Spark with the [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) kit applied: PyPI's `onnxruntime` silently overwrites Jay0515's `sm_121` GPU wheel (same `top_level` import path, different distribution names, no pip conflict detection), and `pip install sageattention` overwrites the local rebuild against current torch/CUDA.

- Shadowing mechanism traced to shared `top_level.txt` declaration across distinct distributions
- `get_available_providers()` is the reliable detector — startup-log GPU-discovery warnings are misleading
- Both findings observed on a live ComfyUI install on 2026-05-22
- Cross-references `fp4_SASS/` for the sm_120 / sm_121 SASS-equivalence basis behind the `sm_120` rebuild target

### 9. [recipes/](recipes/) — Serving launch YAMLs (Spark Arena + our boxes)

**Dated:** leaderboard figures as listed 2026-06-05; local measurements 2026-06-06; vLLM v0.22.1 image. Leaderboard rankings and recipes change over time.

Spark Arena + local measured recipes (NVFP4 and related). Living “run this” counterpart to [`nvfp4/`](nvfp4/). The leaderboard recipes are the submitters' work and are linked, not copied (rank and throughput quoted as listed on 2026-06-05); only our own measured recipes are in this folder.

### 10. [flightrec/](flightrec/) — GB10 flight recorder: run validity, memory guard and monitoring

**Dated:** protocol notes and calibrations from 2026-06-06 to 2026-06-19; added to this repository 2026-08-06; documentation updated 2026-10-23.

Machine-state recorder and safety tooling for DGX Spark (GB10) work, built around repeated out-of-memory freezes and throttled runs polluting results.

- **Run record and validity:** wraps a benchmark or job and stores GPU power, clock, throttle state and energy, CPU load and memory pressure on one timeline. Runs that throttled are marked invalid so they are not mixed into results.
- **Memory guard (`mem_guard`):** opt-in polling daemon that watches `MemAvailable` on the unified-memory pool and warns, then ends runaway model or compute processes, before an out-of-memory condition freezes the machine. A systemd unit is provided. Thresholds are set per machine and need tuning; too tight a setting can end a job that would have finished.
- **Monitoring:** `smoke` (short pre-flight scan), `watch` (stall detection), `live` (throttle alarm).
- **Statistics:** adaptive-N sampling and gates for benchmark results.

- Install: `uv pip install -e ".[dev]"` from `flightrec/`
- Primary target: DGX Spark (GB10); used here for OOM and power-path throttle triage
- Protocols: [`flightrec/PROTOCOLS.md`](flightrec/PROTOCOLS.md)

## Hardware

| | March 2026 lab notes | June 2026 baseline (nvfp4 pass) |
|---|---|---|
| System | NVIDIA DGX Spark (GB10) | same |
| GPU | SM 12.1, 128 GB unified | same |
| CPU | 20-core ARM Grace (aarch64) | same |
| Driver / CUDA | 580.142 · 13.0/13.2 toolkit | **580.159.03 · 13.0.2** (DGX OS 7.5.0) |

Stamp driver · CUDA · image tag · model ID on any new measurement.

## Status

This is work on a single hardware configuration mostly. Results may not generalize to other setups. The TurboQuant container applies the patch from vLLM PR #38479 as it stood when tested; GitHub lists that PR as merged (checked 2026-10-24), and the merged version may differ from the tested patch. Sharing what worked in case it helps others with similar hardware.

Related: [llm-latent-bridge-negative-results](https://github.com/Sggin1/llm-latent-bridge-negative-results) — negative results on bridging LLM hidden states at shallow layers, measured on this hardware.

## License and credits

Original text, code and measurements in this repository are released under the MIT License (see `LICENSE`). Some included files are third-party material under their own licenses (Apache-2.0 for the TurboQuant files taken from vLLM PR #38479, AGPL-3.0 for the Atlas research folder, as its own README states), and several folders are based on other projects, and some link to public leaderboards. `NOTICE.md` lists them with sources and licenses.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Community vLLM container with prebuilt SM121 wheels
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant attention backend by vibhavagarwal5
- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [turboquant-torch](https://pypi.org/project/turboquant-torch/) — Community PyTorch reimplementation
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- The DGX Spark community on NVIDIA Developer Forums
- [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) — Pure Rust+CUDA inference engine, AGPL-3.0
- Atlas Discord community — collaborative benchmarking + maintainer responsiveness
- [eugr's spark-vllm-docker networking deep-wiki](https://deepwiki.com/eugr/spark-vllm-docker/7-dgx-spark-networking) — basis for the dual-spark §2 NCCL config

---

*Topics span March–August 2026 — DGX Spark GB10 / SM121. See each folder for stamped baselines.*
