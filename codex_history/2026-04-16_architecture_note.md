# 2026-04-16 Architecture Note

- What was tried:
  - Inspected `src/alpamayo1_5/models/base_model.py`, `src/alpamayo1_5/models/alpamayo1_5.py`, and `src/alpamayo1_5/diffusion/flow_matching.py` to classify Alpamayo's conditioning path.
- Why it was tried:
  - Needed to answer whether Alpamayo's diffusion stack is "Transformer diffusion" in the sense of using Qwen blocks as the denoiser and whether conditioning is prefix/self-attention or explicit cross-attention.
- What failed:
  - A simple grep for `cross-attention` was misleading because comments mention cross-attention semantics while the actual implementation reuses `past_key_values` in a decoder-style transformer.
- Why it failed:
  - The conditioning mechanism is implemented through cached prefix attention, not through a separately named cross-attention module.
- What succeeded:
  - Confirmed that the VLM backbone is `Qwen3VLForConditionalGeneration`.
  - Confirmed that the diffusion expert is initialized from a copy of `self.vlm.config.text_config` via `AutoModel.from_config(...)`.
  - Confirmed that denoising feeds projected action tokens as `inputs_embeds` into `self.expert(...)` with `past_key_values=prompt_cache` and a custom attention mask, so diffusion tokens attend to the VLM-generated prefix through the transformer self-attention stack.
- Why it succeeded:
  - The relevant code paths are explicit in the model construction and denoiser step.
- Files created or changed:
  - Created `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - Read-only inspection with `rg` and `sed`; no runtime execution.
- Next recommended step:
  - If a deeper architectural write-up is needed, inspect the underlying Hugging Face `Qwen3VL`/`Qwen3` attention implementation to map this cached-prefix path to the exact attention kernel behavior.

## Follow-up: Cross-Attention Alternative

- What was tried:
  - Evaluated the alternative design where VLM-provided KV cache is treated as fixed conditioning memory and diffusion noise tokens act only as queries into a dedicated cross-attention block.
- Why it was tried:
  - Needed to compare this design against Alpamayo's current cached-prefix self-attention expert path.
- What failed:
  - No code change was attempted because the current expert is a decoder-style Qwen block stack, not a module with a ready-made separate cross-attention path.
- Why it failed:
  - Converting to true cross-attention would require architectural surgery: new attention projections, block wiring, mask semantics, cache API changes, and likely retraining or substantial finetuning.
- What succeeded:
  - Identified the main tradeoff: explicit cross-attention could make the conditioning interface cleaner and cheaper to reason about, but it removes token-token mixing between diffusion tokens and the prefix that the current self-attention path gets "for free" inside the same transformer stack.
- Why it succeeded:
  - The current denoiser API already shows that conditioning is implemented as shared-attention over cached prefix plus newly appended action tokens.
- Files created or changed:
  - Updated `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - Conceptual comparison only; no runtime execution.
- Next recommended step:
  - If experimenting, prototype a small adapter block that adds explicit cross-attention after or between Qwen self-attention layers instead of replacing the current path wholesale.

## Follow-up: Training Data Provenance

- What was tried:
  - Checked the local repository README and the public Hugging Face model card for `nvidia/Alpamayo-1.5-10B`.
- Why it was tried:
  - Needed to verify whether Alpamayo 1.5 was trained only on public datasets or also on proprietary/internal data.
- What failed:
  - The local repo README does not spell out the full training-data mixture.
- Why it failed:
  - The repository README focuses on setup and inference, and redirects detailed model information to the Hugging Face model card.
- What succeeded:
  - Confirmed from the Hugging Face model card that Alpamayo 1.5 training data includes public driving datasets and `NVIDIA's internal proprietary autonomous driving data`.
- Why it succeeded:
  - The model card has an explicit "Training Dataset" section describing the data mixture.
- Files created or changed:
  - Updated `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - Read-only local inspection plus public model-card lookup.
- Next recommended step:
  - When discussing fairness of downstream adaptation experiments, treat the released model as already containing knowledge from both public and proprietary pretraining data unless a narrower checkpoint is released.

## Follow-up: Diffusion Layer Identification

- What was tried:
  - Traced `Alpamayo1_5.__init__`, the released model `config.json`, and exported HF configs to identify which module actually carries the diffusion denoiser layers.
- Why it was tried:
  - Needed to locate where a cross-attention replacement would need to be inserted.
- What failed:
  - Treating `self.diffusion` as the main denoiser stack was incorrect.
- Why it failed:
  - `self.diffusion` is only the flow-matching sampler/integrator, while the transformer stack lives in `self.expert`.
- What succeeded:
  - Confirmed that `self.expert = AutoModel.from_config(expert_config)` is the denoiser backbone.
  - Confirmed that `expert_config` starts from `self.vlm.config.text_config` and only width-related fields are overridden by `expert_cfg`.
  - Confirmed from the released config/exported HF configs that the expert keeps `num_hidden_layers=36` while shrinking width from the VLM text tower to `hidden_size=2048`, `num_attention_heads=16`, `intermediate_size=8256`.
  - Confirmed that denoising runs as `action_in_proj -> expert(...) -> action_out_proj`, and `FlowMatching.sample(...)` repeatedly calls this step function.
- Why it succeeded:
  - The model constructor and exported configs make the separation between sampler and transformer denoiser explicit.
- Files created or changed:
  - Updated `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - Read-only inspection with `rg`, `sed`, `nl`, and `python3` on local config files.
- Next recommended step:
  - If modifying conditioning, operate on the 36 Qwen expert layers rather than on `FlowMatching`; e.g. add/replace attention paths inside the expert block stack.

## Follow-up: Where to Insert Cross-Attention

- What was tried:
  - Inspected exported expert weight keys and the installed Hugging Face `transformers` Qwen3 source to determine whether Alpamayo's expert blocks are directly accessible.
- Why it was tried:
  - Needed the exact module path where a new cross-attention block should be inserted.
- What failed:
  - Importing `transformers` from the default shell Python failed.
- Why it failed:
  - The default interpreter was outside the project/venv environment.
- What succeeded:
  - Confirmed from weight keys that the expert block path is `model.layers.<idx>.*` with submodules `self_attn`, `input_layernorm`, `post_attention_layernorm`, and `mlp`.
  - Confirmed from `transformers/models/qwen3/modeling_qwen3.py` that `AutoModel.from_config(Qwen3Config)` builds `Qwen3Model`, whose `self.layers` is a `ModuleList[Qwen3DecoderLayer]`.
  - Confirmed the block order is:
    1. `input_layernorm`
    2. `self_attn`
    3. residual add
    4. `post_attention_layernorm`
    5. `mlp`
    6. residual add
  - Concluded that the natural insertion point for a new cross-attention block is inside each selected `Qwen3DecoderLayer`, most naturally after the self-attention residual and before the MLP.
- Why it succeeded:
  - The HF source exposes both the layer container and the exact decoder-layer forward order.
- Files created or changed:
  - Updated `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - Read-only inspection of local weight maps and local `transformers` source.
- Next recommended step:
  - Implement a custom decoder layer wrapper that replaces selected entries in `self.expert.layers` and adds `cross_attn(+norm)` after `self_attn`, while accepting separate conditioning memory instead of reusing `past_key_values` as self-attention prefix.

## Follow-up: Initial KV-Cache Cross-Attention Patch

- What was tried:
  - Added a wrapper layer that augments selected `self.expert.layers[i]` with a new cross-attention block using diffusion hidden states as queries and the VLM KV cache at the same layer index as keys/values.
- Why it was tried:
  - Needed the first concrete implementation step toward replacing or studying prefix conditioning with explicit cross-attention.
- What failed:
  - No runtime integration test was run against a full Alpamayo inference path.
- Why it failed:
  - The current step focused on code insertion and syntax validation only.
- What succeeded:
  - Added `ExpertKVCacheCrossAttention` implemented with `scaled_dot_product_attention`.
  - Added `Qwen3CrossAttentionDecoderLayer`, which inserts `RMSNorm -> cross_attn -> residual` between the original self-attention residual and the MLP.
  - Added `attach_expert_cross_attention(...)` to replace selected entries in `self.expert.layers`.
  - Added config field `expert_cross_attention_layers` so the patch is opt-in.
- Why it succeeded:
  - Qwen3 expert blocks are directly accessible as a `ModuleList`, so selected layers can be replaced without rewriting the whole expert model.
- Files created or changed:
  - Created `src/alpamayo1_5/models/expert_cross_attention.py`
  - Modified `src/alpamayo1_5/models/alpamayo1_5.py`
  - Modified `src/alpamayo1_5/config.py`
- Validation run:
  - `python3 -m py_compile /home/jys/alpamayo1.5/src/alpamayo1_5/models/expert_cross_attention.py /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py /home/jys/alpamayo1.5/src/alpamayo1_5/config.py`
- Next recommended step:
  - Run a real expert forward/inference smoke test with `expert_cross_attention_layers` enabled to verify cache head shapes, mask semantics, and memory overhead on actual Alpamayo data.

## Follow-up: CLI Wiring for Inference

- What was tried:
  - Added a CLI flag to `src/alpamayo1_5/test_inference.py` to pass cross-attention layer selections into `Alpamayo1_5.from_pretrained(...)`.
- Why it was tried:
  - Needed an easy way to toggle the new expert cross-attention path without editing the script each time.
- What failed:
  - No end-to-end inference run was executed yet.
- Why it failed:
  - This step focused on script wiring and syntax validation only.
- What succeeded:
  - Added `--expert-cross-attention-layers`.
  - Supports `"all"` or comma-separated layer indices such as `"24,28,32,35"`.
  - Parsed CLI values are forwarded to the new `expert_cross_attention_layers` config field.
- Why it succeeded:
  - `from_pretrained(...)` already forwards extra config kwargs cleanly into the model config.
- Files created or changed:
  - Modified `src/alpamayo1_5/test_inference.py`
  - Updated `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - `python3 -m py_compile /home/jys/alpamayo1.5/src/alpamayo1_5/test_inference.py /home/jys/alpamayo1.5/src/alpamayo1_5/models/expert_cross_attention.py /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py /home/jys/alpamayo1.5/src/alpamayo1_5/config.py`
- Next recommended step:
  - Run `test_inference.py --expert-cross-attention-layers ...` on at least one layer subset and inspect for cache-shape/runtime issues before changing the architecture further.

## Follow-up: Prefix Replacement Semantics

- What was tried:
  - Changed the cross-attention-enabled expert wrapper so selected blocks can stop using the VLM prefix cache inside self-attention and rely on cross-attention for conditioning instead.
- Why it was tried:
  - The research direction is to study explicit cross-attention conditioning rather than keeping prefix-conditioned self-attention plus an extra attention path.
- What failed:
  - No numerical or runtime validation was performed yet.
- Why it failed:
  - The work in this step was limited to implementing the routing change and exposing a control flag.
- What succeeded:
  - In selected blocks, self-attention now defaults to local diffusion-token self-attention only (`past_key_values=None` for self-attn).
  - The VLM cache is still passed to the new cross-attention block as conditioning memory.
  - Added config/CLI override to keep the old prefix self-attention if needed for ablation (`--keep-prefix-self-attention`).
- Why it succeeded:
  - The wrapper can independently choose what to pass into the original self-attention and the new cross-attention modules.
- Files created or changed:
  - Modified `src/alpamayo1_5/models/expert_cross_attention.py`
  - Modified `src/alpamayo1_5/config.py`
  - Modified `src/alpamayo1_5/models/alpamayo1_5.py`
  - Modified `src/alpamayo1_5/test_inference.py`
- Validation run:
  - `python3 -m py_compile /home/jys/alpamayo1.5/src/alpamayo1_5/models/expert_cross_attention.py /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py /home/jys/alpamayo1.5/src/alpamayo1_5/config.py /home/jys/alpamayo1.5/src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run a one-layer smoke test first, then compare latency and inspect whether the local self-attention mask semantics are sufficient for the chosen `expert_non_causal_attention` setting.

## Follow-up: Pretrained Weight Loading Mismatch

- What was tried:
  - Investigated the `from_pretrained(...)` warning that all wrapped expert weights were "unused" while `base_layer.*` weights were "newly initialized".
- Why it was tried:
  - Fine-tuning requires preserving the existing expert initialization as much as possible; otherwise the experiment degenerates into near-random reinitialization.
- What failed:
  - Wrapping `expert.layers[i]` directly changed parameter names from `expert.layers.i.*` to `expert.layers.i.base_layer.*`, so pretrained checkpoints no longer matched.
- Why it failed:
  - Hugging Face loading matches parameter names literally and does not know how to redirect the old expert keys into the wrapped module.
- What succeeded:
  - Added a load-time state-dict remapping path:
    - `expert.layers.<i>.* -> expert.layers.<i>.base_layer.*`
    - only for the selected cross-attention layers.
  - Hooked this remapping into `Alpamayo1_5.load_state_dict(...)`.
- Why it succeeded:
  - The mismatch was purely a naming/path issue, so remapping the incoming checkpoint keys is sufficient to preserve the pretrained expert initialization while still allowing new cross-attention parameters to initialize separately.
- Files created or changed:
  - Modified `src/alpamayo1_5/models/expert_cross_attention.py`
  - Modified `src/alpamayo1_5/models/alpamayo1_5.py`
- Validation run:
  - `python3 -m py_compile /home/jys/alpamayo1.5/src/alpamayo1_5/models/expert_cross_attention.py /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`
- Next recommended step:
  - Re-run `from_pretrained(...)` and confirm that only the new cross-attention weights remain newly initialized, while wrapped `base_layer.*` weights are loaded from the checkpoint.

## Follow-up: Diffusion Latency Measurement

- What was tried:
  - Added explicit latency measurement for `vlm_generate` and `diffusion_sample`, while keeping the existing end-to-end timing in the test script.
- Why it was tried:
  - Needed to compare prefix conditioning and cross-attention conditioning on the actual diffusion/planning portion of inference, not only full end-to-end latency.
- What failed:
  - No direct baseline-vs-cross-attention benchmark table was produced yet.
- Why it failed:
  - This step only added instrumentation and reporting.
- What succeeded:
  - Added CUDA-event timing helper inside `Alpamayo1_5`.
  - Measured and returned `vlm_generate` and `diffusion_sample` timings from `sample_trajectories_from_data_with_vlm_rollout(...)` when `return_timings=True`.
  - Updated `test_inference.py` to print:
    - e2e latency
    - VLM latency
    - diffusion latency
    - diffusion/e2e ratio
  - Summary output now averages these metrics across multi-run CLI execution.
- Why it succeeded:
  - The model already has clear `vlm_generate` and `diffusion_sample` boundaries, so CUDA events could be inserted without changing model behavior.
- Files created or changed:
  - Modified `src/alpamayo1_5/models/alpamayo1_5.py`
  - Modified `src/alpamayo1_5/test_inference.py`
- Validation run:
  - `python3 -m py_compile /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py /home/jys/alpamayo1.5/src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run the same clip/time windows once with default prefix conditioning and once with `--expert-cross-attention-layers all`, then compare mean diffusion latency and diffusion/e2e ratio over the same `--nums` range.

## Training plan note
- What was tried:
  - Re-checked local code to determine what is needed to train the cross-attention variant from the current `alpamayo1.5` repo state.
  - Verified that local repo does not contain the previously discussed upstream `finetune/` directory, so training must be built around the patched local model code.
- Why it was tried:
  - The user asked how to train the current cross-attention replace variant.
- What failed:
  - Looking for `/home/jys/alpamayo1.5/finetune` locally.
- Why it failed:
  - The local checkout does not include that directory.
- What succeeded:
  - Confirmed the current patch supports the right initialization semantics for fine-tuning: pretrained expert weights load into `base_layer.*`, while only `cross_attn.*` and `cross_attn_layernorm.*` are newly initialized.
  - Confirmed current replace semantics: when `expert_cross_attention_layers` is set, all expert layers stop using prefix self-attention; only selected layers perform cross-attention to VLM KV cache.
- Why it succeeded:
  - The wrapper/remap implementation is already in place in `expert_cross_attention.py` and `alpamayo1_5.py`.
- Files created or changed:
  - Updated this history note only.
- Validation run:
  - Code inspection with `sed` on `expert_cross_attention.py`, `alpamayo1_5.py`, and `load_physical_aiavdataset.py`.
- Next recommended step:
  - Build a minimal fine-tuning script around the current model that freezes everything except selected `cross_attn.*` modules (and optionally nearby norms), uses the PhysicalAI dataset loader for samples, and optimizes trajectory loss against `ego_future_xyz` / `ego_future_rot`.

## Nsight profiling note
- What was tried:
  - Checked current NVTX coverage for the VLM-side inference path and verified `nsys` availability.
- Why it was tried:
  - The user wants to profile the VLM side with Nsight Systems.
- What failed:
  - No code failure; current NVTX coverage was already sufficient for a first VLM-side trace.
- Why it failed:
  - N/A.
- What succeeded:
  - Confirmed existing NVTX ranges around `fuse_traj_tokens`, `vlm_generate`, `postprocess_generate_outputs`, and outer inference/script stages.
  - Confirmed `nsys` exists at `/usr/local/cuda/bin/nsys`.
- Why it succeeded:
  - The inference script and model already contain explicit `nvtx_range(...)` annotations.
- Files created or changed:
  - Updated this history note only.
- Validation run:
  - `which nsys`
  - `rg -n "nvtx_range\(|vlm_generate" src/alpamayo1_5/models/alpamayo1_5.py`
- Next recommended step:
  - Run `nsys profile` on `src/alpamayo1_5/test_inference.py`, then inspect the timeline around `vlm_generate` first and compare prefix vs cross-attention runs separately.

## Nsight Compute note
- What was tried:
  - Checked whether Nsight Compute CLI is available for compute-level profiling of the `vlm_generate` NVTX range.
- Why it was tried:
  - The user wants more detailed compute analysis than Nsight Systems provides.
- What failed:
  - Nothing failed.
- Why it failed:
  - N/A.
- What succeeded:
  - Confirmed `ncu` exists and is usable.
- Why it succeeded:
  - The CUDA toolkit install includes Nsight Compute CLI.
- Files created or changed:
  - Updated this history note only.
- Validation run:
  - `which ncu`
  - `ncu --version`
- Next recommended step:
  - Profile only the `vlm_generate` NVTX range with `ncu --nvtx --nvtx-include vlm_generate` and start from `--set full` or a targeted roofline/speed-of-light section set.

## VLM-only profiling path
- What was tried:
  - Added a `--vlm-only` execution path to `test_inference.py` so the script can stop after `self.vlm.generate(...)` and skip diffusion entirely.
- Why it was tried:
  - The user wants to profile only the VLM compute path with Nsight Compute.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Implemented a VLM-only path that preserves the same preprocessing and `fuse_traj_tokens`, then runs `vlm.generate`, postprocesses generated sequences, and prints CoC plus VLM-only latency.
- Why it succeeded:
  - The existing inference script already contained all required pieces; only a branch was needed.
- Files created or changed:
  - Modified `src/alpamayo1_5/test_inference.py`.
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Use `python3 src/alpamayo1_5/test_inference.py --vlm-only` for quick validation, then run `ncu` on that exact path to capture only VLM kernels.

## NCU permission diagnosis
- What was tried:
  - Checked why `ncu` still reported no kernels while VLM-only inference completed.
- Why it was tried:
  - The user reported `ERR_NVGPUCTRPERM` and empty Nsight Compute output.
- What failed:
  - `ncu` profiling itself.
- Why it failed:
  - NVIDIA driver parameter `RmProfilingAdminOnly` is set to `1`, so non-admin users cannot access GPU performance counters.
- What succeeded:
  - Confirmed the failure is environmental, not in the Alpamayo code path.
- Why it succeeded:
  - `/proc/driver/nvidia/params` exposes `RmProfilingAdminOnly: 1`.
- Files created or changed:
  - Updated this history note only.
- Validation run:
  - `cat /proc/driver/nvidia/params`
  - `nvidia-smi -q`
- Next recommended step:
  - Either run `ncu` with admin privileges, or change the NVIDIA driver profiling permission setting (e.g. `NVreg_RestrictProfilingToAdminUsers=0`) and reload/reboot before re-running Nsight Compute.

## Porting strategy note
- What was tried:
  - Planned the minimum file import/edit strategy for reusing NVlabs/alpamayo training code with local Alpamayo1.5 cross-attention experiments.
- Why it was tried:
  - The user asked which files to bring over and modify, instead of copying the entire upstream training repo blindly.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Identified that `finetune/` alone is insufficient because upstream training code depends on `src/alpamayo_r1` modules and package layout.
  - Narrowed the practical path to reusing only the SFT training harness pieces while implementing local `alpamayo1_5`-specific model and dataset glue.
- Why it succeeded:
  - Upstream raw files show direct imports from `alpamayo_r1.*` and `finetune.sft.*`, while local repo already contains the patched inference/model code needed for Alpamayo1.5.
- Files created or changed:
  - Updated this history note only.
- Validation run:
  - Manual inspection of upstream raw GitHub files and local patched model files.
- Next recommended step:
  - Create a new local training subtree under `alpamayo1.5/finetune_local/` with: (1) trainer entry/config copied from upstream SFT, (2) a local trainable Alpamayo1.5 model wrapper, and (3) a nuScenes dataset adapter that emits Alpamayo-style tensors.

## Selective prefix path
- What was tried:
  - Added a training-free selective-prefix attention path so only chosen expert blocks consume VLM `past_key_values` while other blocks run local self-attention on diffusion tokens only.
- Why it was tried:
  - The user wanted a `--select-prefix` option to test whether only conditioning-sensitive layers need prefix attention.
- What failed:
  - `python3 ... --help` with the system Python failed due to missing `numpy`.
- Why it failed:
  - The non-venv Python environment does not have the project dependencies installed.
- What succeeded:
  - Added config support via `expert_prefix_layers`.
  - Extended expert wrapping so wrapping is activated when either cross-attn layers or prefix-selected layers are requested.
  - Selected prefix layers keep `past_key_values`; unselected wrapped layers ignore prefix KV.
  - Added CLI option `--select-prefix` to `test_inference.py`.
- Why it succeeded:
  - The existing wrapper already controlled whether each layer uses prefix self-attention; only per-layer selection wiring was needed.
- Files created or changed:
  - `src/alpamayo1_5/config.py`
  - `src/alpamayo1_5/models/expert_cross_attention.py`
  - `src/alpamayo1_5/models/alpamayo1_5.py`
  - `src/alpamayo1_5/test_inference.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/config.py src/alpamayo1_5/models/expert_cross_attention.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run inference under the venv with `--select-prefix` for a few sparse settings (e.g. 3, 6, 12 layers) and compare diffusion latency / minADE against the full-prefix baseline.

## Prefix-only wrapper fix
- What was tried:
  - Split the selective-prefix path from the cross-attention wrapper after observing that `--select-prefix` still instantiated `cross_attn_layernorm` parameters and loaded as if cross-attention were present.
- Why it was tried:
  - The user correctly pointed out that prefix-selection and cross-attention are different experiment paths and should not share parameterized cross-attention modules.
- What failed:
  - The initial `--select-prefix` implementation reused `Qwen3CrossAttentionDecoderLayer`, which created spurious `cross_attn_layernorm` weights even when cross-attention was disabled.
- Why it failed:
  - The wrapper class always instantiated the cross-attention norm path.
- What succeeded:
  - Added `Qwen3SelectivePrefixDecoderLayer` with no cross-attention parameters.
  - Updated `attach_expert_cross_attention(...)` so prefix-only runs use the new wrapper, while cross-attention runs still use the original cross-attention wrapper.
- Why it succeeded:
  - Prefix selection only needs per-layer control over whether `past_key_values` are forwarded to self-attention; it does not need cross-attention modules.
- Files created or changed:
  - `src/alpamayo1_5/models/expert_cross_attention.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/models/expert_cross_attention.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py src/alpamayo1_5/config.py`
- Next recommended step:
  - Re-run `python3 src/alpamayo1_5/test_inference.py --select-prefix 1,3,5 --nums 1` inside the venv and confirm there are no `cross_attn_*` newly-initialized warnings before measuring latency/accuracy.

## Block attention workload debug
- What was tried:
  - Added an optional debug path to print per-block attention workload estimates during inference.
- Why it was tried:
  - The user wanted to verify that selective-prefix runs are truly reducing attention work in non-selected layers.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Added a lightweight capture mechanism in `expert_cross_attention.py` for the first expert forward of each inference.
  - Captured, per layer: whether prefix KV is used, query length, KV length, number of heads, and estimated attention score count (`q * kv * heads`).
  - Exposed this through `return_attention_debug` and `--print-block-attn-stats` in `test_inference.py`.
- Why it succeeded:
  - Attention shape information is available directly inside the wrapped expert layers.
- Files created or changed:
  - `src/alpamayo1_5/models/expert_cross_attention.py`
  - `src/alpamayo1_5/models/alpamayo1_5.py`
  - `src/alpamayo1_5/src/alpamayo1_5/test_inference.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/models/expert_cross_attention.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run `python3 src/alpamayo1_5/test_inference.py --select-prefix 1,3,5 --nums 1 --print-block-attn-stats` inside the venv and confirm only layers 1, 3, and 5 show large KV lengths while all other layers show local-only KV length equal to the diffusion token count.

## File rename cleanup
- What was tried:
  - Renamed the helper module that previously bundled both cross-attention and selective-prefix logic.
- Why it was tried:
  - The old filename `expert_cross_attention.py` had become misleading once the selective-prefix experiment was added.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Renamed `src/alpamayo1_5/models/expert_cross_attention.py` to `src/alpamayo1_5/models/expert_selective_layer.py` and updated imports.
- Why it succeeded:
  - The module now serves as a general layer-wise conditioning control utility, not only cross-attention.
- Files created or changed:
  - Renamed `src/alpamayo1_5/models/expert_cross_attention.py` -> `src/alpamayo1_5/models/expert_selective_layer.py`
  - Updated `src/alpamayo1_5/models/alpamayo1_5.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py`
- Next recommended step:
  - If the selective-prefix path remains the main direction, consider renaming helper function names and config keys in a second cleanup pass; keep current names for now to avoid breaking CLI/config compatibility.

## Git rollback checkpoint
- What was tried:
  - Verified whether current selective-prefix and profiling instrumentation work had been recorded in git, then updated repository instructions to require coherent checkpoint commits for rollback.
- Why it was tried:
  - The user explicitly asked that rollback be possible at any time and that this policy be reflected in `AGENTS.md`.
- What failed:
  - The current working set had not yet been committed, even though history notes were being maintained.
- Why it failed:
  - Work had been accumulating in the working tree without a dedicated checkpoint commit.
- What succeeded:
  - Clarified the git checkpoint requirement in `AGENTS.md` and prepared the current work to be committed as a rollback point.
- Why it succeeded:
  - The repository already had an AGENTS policy and a coherent set of related code changes for selective prefix / profiling instrumentation.
- Files created or changed:
  - `AGENTS.md`
  - current selective-prefix/profiling code files
  - this history note
- Validation run:
  - `git -C /home/jys/alpamayo1.5 status --short`
- Next recommended step:
  - Commit only the relevant source/history files for this checkpoint and keep profiler artifacts or unrelated experimental directories out of the commit.

## CLI cleanup
- What was tried:
  - Shortened the block-attention debug CLI flag.
- Why it was tried:
  - The user requested a shorter option name.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Renamed `--print-block-attn-stats` to `--print-block` while preserving the same behavior.
- Why it succeeded:
  - The debug path was isolated to a single CLI flag in `test_inference.py`.
- Files created or changed:
  - `src/alpamayo1_5/test_inference.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Use `--print-block --nums 1` when validating selective-prefix layer behavior.

## Selective prefix sweep script
- What was tried:
  - Added a shell script to automate selective-prefix sensitivity sweeps over multiple layer sets.
- Why it was tried:
  - The user wanted a practical experiment script that logs `minADE` and latency results to judge which layer sets are most sensitive.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Created `tools/run_selective_prefix_sweep.sh`.
  - The script runs a fixed list of baseline / grouped / late-layer candidate prefix sets.
  - It stores raw stdout per run under `codex_history/selective_prefix_sweep_<timestamp>/raw/` and writes a summary CSV with mean `minADE`, mean e2e latency, mean diffusion latency, and mean diffusion/e2e ratio.
- Why it succeeded:
  - `test_inference.py` already prints a parseable summary line when `--nums > 1`.
- Files created or changed:
  - `tools/run_selective_prefix_sweep.sh`
  - `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - `bash -n tools/run_selective_prefix_sweep.sh`
- Next recommended step:
  - Run the sweep script inside the venv, inspect the summary CSV for the best group, then refine around the best-performing layer region with a second smaller sweep.

## Sweep runner interpreter fix
- What was tried:
  - Fixed the sweep script's Python interpreter selection after all runs failed with `ModuleNotFoundError: numpy`.
- Why it was tried:
  - The initial script defaulted to a repo-local venv path that did not match the user's active environment.
- What failed:
  - The first sweep run.
- Why it failed:
  - `tools/run_selective_prefix_sweep.sh` invoked a Python interpreter without the project dependencies installed.
- What succeeded:
  - Updated the script to prefer `PYTHON_BIN` if provided, then the active `VIRTUAL_ENV`, then `/home/jys/a1_5_venv/bin/python3`, and only finally the repo-local fallback.
- Why it succeeded:
  - The user's actual interactive runs have been using the external `a1_5_venv`, not the repo-local fallback path.
- Files created or changed:
  - `tools/run_selective_prefix_sweep.sh`
- Validation run:
  - `bash -n tools/run_selective_prefix_sweep.sh`
- Next recommended step:
  - Re-run the sweep script from the activated `a1_5_venv`, or set `PYTHON_BIN=/home/jys/a1_5_venv/bin/python3` explicitly for reproducibility.

## Sweep parser fix
- What was tried:
  - Fixed the sweep summary parser after the script completed runs but marked every case as `missing_summary`.
- Why it was tried:
  - The user's environment did not have `rg`, so the summary extraction step failed even though inference completed and printed summary lines.
- What failed:
  - Parsing the `Summary over ...` line from each raw log.
- Why it failed:
  - The script used `rg`, which was not installed in the shell environment running the sweep.
- What succeeded:
  - Replaced `rg` with `grep` for summary extraction.
- Why it succeeded:
  - `grep` is available by default and sufficient for this simple single-line parse.
- Files created or changed:
  - `tools/run_selective_prefix_sweep.sh`
- Validation run:
  - `bash -n tools/run_selective_prefix_sweep.sh`
- Next recommended step:
  - Re-run the sweep, or re-parse existing raw logs with the fixed script logic if you want the CSV summary without repeating all experiments.

## Selective prefix sweep results
- What was tried:
  - Ran the automated selective-prefix sweep after fixing the interpreter and summary parsing.
- Why it was tried:
  - To estimate which layer groups are relatively more conditioning-sensitive under training-free prefix sparsification.
- What failed:
  - None of the tested sparse/grouped prefix settings preserved baseline accuracy.
- Why it failed:
  - The pretrained model is aligned to full-prefix conditioning, so naively dropping prefix from most layers causes substantial accuracy collapse.
- What succeeded:
  - Produced a valid summary CSV at `codex_history/selective_prefix_sweep_20260416_160050/summary.csv`.
  - Baseline full prefix: mean minADE `1.0337m`, mean diffusion latency `215.90ms`.
  - Best among the tested sparse groups was `early_0_5`: mean minADE `3.4687m`, mean diffusion latency `130.22ms`.
  - `late_24_29` was close: mean minADE `3.7205m`, mean diffusion latency `130.69ms`.
- Why it succeeded:
  - The sweep script now uses the correct Python environment and parses summary lines with `grep`.
- Files created or changed:
  - `codex_history/selective_prefix_sweep_20260416_160050/summary.csv`
  - raw logs under `codex_history/selective_prefix_sweep_20260416_160050/raw/`
- Validation run:
  - `./tools/run_selective_prefix_sweep.sh`
- Next recommended step:
  - Run a second-stage refinement around the least-bad groups (`0-5` and `24-29`) instead of broader sets, and compare contiguous vs interleaved choices nearby.

## Selective prefix refinement sweep
- What was tried:
  - Added a second-stage refinement sweep script focused on the two least-bad groups from the first sweep (`0-5` and `24-29`).
- Why it was tried:
  - The first sweep showed that broad sparse choices were mostly poor, but `0-5` and `24-29` were relatively less damaging, so the next step is local refinement rather than random new groups.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Created `tools/run_selective_prefix_refine.sh` with contiguous, shifted, and interleaved variants around the early and late candidate regions.
  - The script writes raw logs and a summary CSV just like the first sweep.
- Why it succeeded:
  - The first sweep already established a reusable automation pattern and identified concrete regions worth refining.
- Files created or changed:
  - `tools/run_selective_prefix_refine.sh`
- Validation run:
  - `bash -n tools/run_selective_prefix_refine.sh`
- Next recommended step:
  - Run the refinement sweep and compare not just the best mean minADE, but also whether any refined set improves over the `0-5` / `24-29` first-stage candidates without giving back too much latency.

## Selective prefix refinement results
- What was tried:
  - Ran the second-stage refinement sweep around the two least-bad regions from the first sweep: `0-5` and `24-29`.
- Why it was tried:
  - To check whether a nearby contiguous or interleaved sparse layer set could preserve more accuracy without giving back too much latency.
- What failed:
  - No refined sparse set came close to the full-prefix baseline accuracy.
- Why it failed:
  - The pretrained model appears strongly aligned to full-prefix conditioning, so even the best sparse choices still lose substantial planning quality.
- What succeeded:
  - Produced a valid refinement summary at `codex_history/selective_prefix_refine_20260416_165807/summary.csv`.
  - Full prefix baseline: mean minADE `1.0337m`, diffusion latency `217.43ms`.
  - Best refined sets were early-layer biased:
    - `0,1,4,5`: mean minADE `3.4109m`, diffusion latency `125.18ms`
    - `0,2,4,6,8`: mean minADE `3.4138m`, diffusion latency `127.14ms`
    - `0,1,2,3,4,5`: mean minADE `3.4687m`, diffusion latency `130.02ms`
  - Best late-layer refined set was `24,25,26,27,28,29`: mean minADE `3.7205m`, diffusion latency `130.98ms`.
- Why it succeeded:
  - The refinement sweep focused on the only regions that were not obviously catastrophic in the first pass.
- Files created or changed:
  - `codex_history/selective_prefix_refine_20260416_165807/summary.csv`
  - raw logs under `codex_history/selective_prefix_refine_20260416_165807/raw/`
- Validation run:
  - `./tools/run_selective_prefix_refine.sh`
- Next recommended step:
  - If continuing this direction, frame the current finding as `training-free sparse prefix dropping gives strong latency gains but still significant quality loss`; further work would need either learned sensitivity metrics or retraining/fine-tuning to recover accuracy.

## Head and dim ablation hooks
- What was tried:
  - Added single-head and single-dimension ablation support to expert self-attention, plus sweep scripts for head-level and dim-level experiments.
- Why it was tried:
  - The user wants to measure sensitivity not just at the layer level, but at the head level and eventually the per-dimension level within each head.
- What failed:
  - N/A.
- Why it failed:
  - N/A.
- What succeeded:
  - Added config/CLI support for `--ablate-head <layer>:<head>` and `--ablate-dim <layer>:<head>:<dim>`.
  - Implemented forward-time ablation by temporarily zeroing the corresponding input columns of `self_attn.o_proj`, which removes the contribution of the chosen head or sub-dimension for one layer.
  - Added `tools/run_head_ablation_sweep.sh` and `tools/run_dim_ablation_sweep.sh` with CSV logging and raw log capture.
- Why it succeeded:
  - Ablating `o_proj` input columns is a clean way to suppress one head/sub-dimension contribution without rewriting the Hugging Face attention kernel.
- Files created or changed:
  - `src/alpamayo1_5/config.py`
  - `src/alpamayo1_5/models/expert_selective_layer.py`
  - `src/alpamayo1_5/models/alpamayo1_5.py`
  - `src/alpamayo1_5/test_inference.py`
  - `tools/run_head_ablation_sweep.sh`
  - `tools/run_dim_ablation_sweep.sh`
- Validation run:
  - `bash -n tools/run_head_ablation_sweep.sh`
  - `bash -n tools/run_dim_ablation_sweep.sh`
  - `python3 -m py_compile src/alpamayo1_5/config.py src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Start with a small head sweep on the best current sparse prefix set (for example `SELECT_PREFIX=0,1,4,5`) before attempting any full 36x16x128 dim sweep, which will be extremely expensive.

## Ablation sweep design refinement
- What was tried:
  - Reworked the head/dim sweep scripts so the experiment design is built in, instead of requiring the user to hand-specify ranges.
- Why it was tried:
  - The user explicitly asked for the sweep scope to be designed in the scripts rather than left as unconstrained brute-force ranges.
- What failed:
  - The first head/dim sweep scripts were too generic and pushed the experimental design burden back onto the user.
- Why it failed:
  - Leaving the full 36x16x128 search space unconstrained is not a realistic default.
- What succeeded:
  - Redesigned `run_head_ablation_sweep.sh` to default to the best current sparse prefix set (`0,1,4,5`) and sweep all 36x16 heads under that context.
  - Redesigned `run_dim_ablation_sweep.sh` to consume the latest head sweep summary and automatically select the top-N most sensitive heads (default 8), then sweep all 128 dims only for those heads.
  - Added baseline rows and `delta_vs_baseline_m` columns to both summaries.
- Why it succeeded:
  - The current best sparse set and the natural hierarchy of `layer -> head -> dim` make a staged search much more tractable and interpretable.
- Files created or changed:
  - `tools/run_head_ablation_sweep.sh`
  - `tools/run_dim_ablation_sweep.sh`
  - `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - `bash -n tools/run_head_ablation_sweep.sh`
  - `bash -n tools/run_dim_ablation_sweep.sh`
  - `python3 -m py_compile src/alpamayo1_5/config.py src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run `tools/run_head_ablation_sweep.sh` first, inspect the top heads by `delta_vs_baseline_m`, then let `tools/run_dim_ablation_sweep.sh` automatically refine only those heads.

## Drop-prefix experiment shift
- What was tried:
  - Reframed the sparse-prefix sensitivity experiments from `keep-only` to `drop-from-full-prefix` and added corresponding CLI/config plus two new sweep scripts.
- Why it was tried:
  - The user pointed out that true sensitivity should be measured by comparing each block with vs. without prefix while keeping all other blocks unchanged.
- What failed:
  - The earlier `keep-only` design mixed block importance with catastrophic context removal, making sensitivity interpretation less direct.
- Why it failed:
  - Keeping only a small set of blocks is a much harsher intervention than ablating one block/group from the full-prefix baseline.
- What succeeded:
  - Added `expert_drop_prefix_layers` and `--drop-prefix`.
  - Updated selective-layer wiring so drop-prefix removes prefix only on chosen layers while all other layers remain full-prefix.
  - Added `tools/run_drop_prefix_sweep.sh` for group ablations from the full baseline.
  - Added `tools/run_drop_prefix_refine.sh` for second-stage local refinement around the least-sensitive groups.
- Why it succeeded:
  - Drop-prefix is the correct direct measure of which blocks are insensitive to removing prefix conditioning.
- Files created or changed:
  - `src/alpamayo1_5/config.py`
  - `src/alpamayo1_5/models/expert_selective_layer.py`
  - `src/alpamayo1_5/models/alpamayo1_5.py`
  - `src/alpamayo1_5/test_inference.py`
  - `tools/run_drop_prefix_sweep.sh`
  - `tools/run_drop_prefix_refine.sh`
- Validation run:
  - `bash -n tools/run_drop_prefix_sweep.sh`
  - `bash -n tools/run_drop_prefix_refine.sh`
  - `python3 -m py_compile src/alpamayo1_5/config.py src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run `tools/run_drop_prefix_sweep.sh` first; only after identifying the least-sensitive dropped group should the refinement sweep or head/dim ablations be interpreted seriously.

## Automatic full drop-prefix exploration
- What was tried:
  - Upgraded the drop-prefix sweep design so the first-stage script automatically explores the entire layer space instead of relying on hand-picked drop candidates.
- Why it was tried:
  - The user correctly pointed out that a proper first-pass sensitivity study should automatically test all blocks, not just manually chosen groups.
- What failed:
  - The previous drop-prefix sweep still depended on curated candidate groups.
- Why it failed:
  - It did not perform an exhaustive first-pass search over all 36 blocks.
- What succeeded:
  - `run_drop_prefix_sweep.sh` now runs:
    - full-prefix baseline
    - all 36 single-layer drop-prefix ablations
    - coarse contiguous group drops (`GROUP_SIZE`, default 6)
  - `run_drop_prefix_refine.sh` now automatically reads the latest sweep summary and reruns the least-sensitive single layers and groups instead of relying on hard-coded candidates.
- Why it succeeded:
  - Single-layer drop ablations are the direct measure of block sensitivity, and group drops provide coarse region-level context in the same pass.
- Files created or changed:
  - `tools/run_drop_prefix_sweep.sh`
  - `tools/run_drop_prefix_refine.sh`
  - `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - `bash -n tools/run_drop_prefix_sweep.sh`
  - `bash -n tools/run_drop_prefix_refine.sh`
  - `python3 -m py_compile src/alpamayo1_5/config.py src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Run `tools/run_drop_prefix_sweep.sh` to identify the least-sensitive dropped blocks, then only interpret head/dim ablations inside blocks that the first-pass full scan deems important.

## Drop-prefix activation bug
- What was tried:
  - Investigated why `--drop-prefix` runs produced baseline-identical metrics and empty `--print-block` self-attention debug output.
- Why it was tried:
  - The user observed that dropping any layer or group had zero effect, which contradicted prior sparse-prefix experiments.
- What failed:
  - `--drop-prefix` did not actually activate the selective-layer wrappers.
- Why it failed:
  - In `attach_expert_cross_attention(...)`, the early-return condition ignored `drop_prefix_indices`, so a pure drop-prefix request returned without wrapping any expert layers.
- What succeeded:
  - Updated the activation condition so `drop_prefix_indices` also triggers wrapper installation.
- Why it succeeded:
  - Drop-prefix experiments only work when expert layers are wrapped and the selected layers can switch from prefix to local self-attention.
- Files created or changed:
  - `src/alpamayo1_5/models/expert_selective_layer.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py src/alpamayo1_5/config.py`
- Next recommended step:
  - Re-run `python3 src/alpamayo1_5/test_inference.py --drop-prefix 0 --nums 1 --print-block` and confirm only layer 0 switches to `mode=local`; then rerun the drop-prefix sweep.

## Attention debug capture fix
- What was tried:
  - Fixed repeated block-attention debug output after confirming `--drop-prefix 0` correctly switched only layer 0 to local attention.
- Why it was tried:
  - The debug print was repeating the same 36-layer report for every diffusion step, making inspection noisy.
- What failed:
  - `capture_once=True` did not actually stop later expert forward captures for self-attention debug.
- Why it failed:
  - The `captured` flag was only being set on the cross-attention debug path, not the self-attention-only path.
- What succeeded:
  - Added an explicit `record_current_call` gate so only the first expert forward is recorded when `capture_once=True`.
- Why it succeeded:
  - The capture state now flips to recorded at the start of the first expert forward instead of depending on cross-attention instrumentation.
- Files created or changed:
  - `src/alpamayo1_5/models/expert_selective_layer.py`
- Validation run:
  - `python3 -m py_compile src/alpamayo1_5/models/expert_selective_layer.py src/alpamayo1_5/models/alpamayo1_5.py src/alpamayo1_5/test_inference.py`
- Next recommended step:
  - Re-run a one-sample `--drop-prefix ... --print-block` command to confirm a single 36-layer report is printed, then rerun the full drop-prefix sweep.

## Block sensitivity status and follow-up automation
- What was tried:
  - Re-ran block sensitivity analysis after fixing drop-prefix activation and attention-debug capture, then added a follow-up sweep script for the next recommended experiments.
- Why it was tried:
  - The previous drop-prefix sweep had been invalid before the wrapper activation fix, and the next step needed to be automated instead of manually running each follow-up case.
- What failed:
  - The first `drop_prefix_sweep_20260417_111601` summary was effectively invalid, and the CSV formatting in later summaries remained messy.
- Why it failed:
  - The early sweep ran before `drop_prefix_indices` actually triggered wrapper installation; later the raw metrics were correct but summary rows still contained duplicated newline-separated values.
- What succeeded:
  - A valid drop-prefix sweep at `codex_history/drop_prefix_sweep_20260417_154623/summary.csv` showed meaningful block-level sensitivity:
    - least-sensitive single-block drops included `34` and `27`
    - highly sensitive blocks clustered around `12~15`
    - least-sensitive coarse group was around `18~23`
  - `python3 src/alpamayo1_5/test_inference.py --drop-prefix 0 --nums 1 --print-block` now cleanly shows only layer `0` switching to `mode=local`, with the other 35 layers remaining `mode=prefix`.
  - Added `tools/run_sensitivity_followups.sh` to automate:
    - rechecks of least-/most-sensitive single-block drops
    - refinements around the relatively insensitive `18~23` region
    - exhaustive head ablations for all `36 x 16 = 576` heads across the full expert stack
- Why it succeeded:
  - The repaired drop-prefix path now measures block sensitivity against the full-prefix baseline correctly, and the follow-up script now covers both the low-impact block search and a full head-level sensitivity map across the entire model.
- Files created or changed:
  - `src/alpamayo1_5/models/expert_selective_layer.py`
  - `tools/run_sensitivity_followups.sh`
  - `codex_history/2026-04-16_architecture_note.md`
- Validation run:
  - `python3 src/alpamayo1_5/test_inference.py --drop-prefix 0 --nums 1 --print-block`
  - `bash -n tools/run_sensitivity_followups.sh`
- Next recommended step:
  - Run `PYTHON_BIN=/home/jys/a1_5_venv/bin/python3 ./tools/run_sensitivity_followups.sh`, then rank all 576 heads by `delta_vs_baseline_m` to identify globally low-impact heads worth pruning first.
