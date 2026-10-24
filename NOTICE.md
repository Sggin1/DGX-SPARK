# Notice: licenses, sources and credits

Last reviewed: 2026-10-24.

## Scope of the repository license

Original text, code, scripts and measurements in this repository are released under the MIT License (`LICENSE`). Three folders carry their own copy of the same MIT terms (`flightrec/LICENSE`, `dual-spark/LICENSE`, `fp4_SASS/LICENSE`). Files that came from other projects keep their own licenses, listed below. Where a file or folder states a different license, that statement applies to it.

## Third-party material included in this repository

| Path | Source | License | Notes |
| --- | --- | --- | --- |
| `turboquant/new_files/` and `turboquant/patches/pr_38479.diff` | [vllm-project/vllm pull request #38479](https://github.com/vllm-project/vllm/pull/38479) (author: vibhavagarwal5, with other contributors) | Apache-2.0 (SPDX headers retained in the files; text in `LICENSES/Apache-2.0.txt`) | The TurboQuant implementation files and the patch were taken from that pull request for testing on DGX Spark. The pull request is credited to its authors; see `turboquant/README.md` for how it was applied. |
| `atlas/omni-fix-proposal/working-tree.diff` | A patch against [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) | AGPL-3.0 (text in `LICENSES/AGPL-3.0.txt`) | A diff contains lines of the Atlas source as context, so it is treated as covered by Atlas's license, not by the MIT license. |
| `recipes/` (the `rank*.yaml` files and the leaderboard tables in `recipes/README.md`) | Recipes listed on the [Spark Arena](https://spark-arena.com/leaderboard) community leaderboard, captured on 2026-06-05 | Not checked | These are the submitters' work and the throughput numbers are theirs. They are included for reference with credit to the submitters and the source. The `measured-*.yaml` files record the repository author's own measurements. If you are an author of a listed recipe and would like it changed or removed, please open an issue. |

## Other projects referenced (not included)

The folders below are built on, run, or discuss these projects. Their code is not copied here. Licenses are as shown on their GitHub pages on 2026-10-24.

| Project | License | Used for |
| --- | --- | --- |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | serving and the TurboQuant test |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and [eugr/llama-benchy](https://github.com/eugr/llama-benchy) | MIT | container recipes (`recipes/`, `turboquant/`) and the benchmark tool named in `dual-spark/` and the recipe headers |
| [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) | Apache-2.0 | referenced in the dual-spark notes |
| [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) | AGPL-3.0 | inference engine benchmarked in `atlas/` |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | `nemo3-super-gguf/` |
| [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) | MIT | the upstream kit discussed in `comfyui_spark_notes/` |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | Apache-2.0 | `mamba-dev/` |
| [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | BSD-3-Clause | `mamba-dev/` |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 | linked in `nvfp4/archive/landscape-2026-03-26.md` |
| [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) and [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Apache-2.0 | linked in `nvfp4/archive/landscape-2026-03-26.md` |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 | linked in `nvfp4/archive/landscape-2026-03-26.md` |
| [hoesing/spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check) | MIT | referenced in `GX10_PD_Throttle_Fix.md` |

Also linked in `nvfp4/archive/landscape-2026-03-26.md`: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (GitHub does not summarise its license; see the project).

## Models, wheels and forum material

- Models and wheels (for example on Hugging Face) are referenced by name and link only. Their licenses are on their model cards and are not changed by this repository.
- Short quotations and links to NVIDIA Developer Forum posts, GitHub issues and pull requests are included for context. Copyright in those texts remains with their authors.

## Corrections and removals

If anything here is credited incorrectly, or you would like material removed or re-licensed, please open an issue on this repository.
