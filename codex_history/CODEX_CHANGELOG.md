# Codex Change Log

This file records every Codex-side change made in this workspace so the additions can be audited and rolled back without touching the original source files.

## 2026-04-02

### Change 001

- Type: Added file
- Path: `src/alpamayo1_5/profiling.py`
- Reason:
  - Added isolated profiling wrappers for component-level capture without editing the original model implementation.
  - Added NVTX range helpers around `vision_encoder`, `vlm_prefill`, `vlm_decode_step`, `expert_forward`, `expert_denoiser_step`, and `diffusion_sample`.
  - Added helper functions to reconstruct generation artifacts and diffusion inputs from a VLM rollout.
- Rollback:
  - Remove the file:
    - `rm /home/jys/alpamayo1.5/src/alpamayo1_5/profiling.py`
- Notes:
  - No existing source file was modified for this change.
  - Future Codex changes should continue to be logged here before or immediately after they are made.

### Change 002

- Type: Modified file
- Path: `src/alpamayo1_5/models/alpamayo1_5.py`
- Reason:
  - Inserted an `nvtx_range()` context manager directly into the original inference path.
  - Added NVTX markers around the major runtime boundaries in `sample_trajectories_from_data_with_vlm_rollout()`.
  - Added nested NVTX markers inside the denoiser step for `action_in_proj`, `expert_forward`, and `action_out_proj`.
- Inserted markers:
  - `fuse_traj_tokens`
  - `vlm_generate`
  - `postprocess_generate_outputs`
  - `diffusion_sample`
  - `denoiser_step`
  - `action_in_proj`
  - `expert_forward`
  - `action_out_proj`
  - `action_to_traj`
  - `format_outputs`
- Rollback:
  - Revert the file to the previous revision in git:
    - `git -C /home/jys/alpamayo1.5 checkout -- /home/jys/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`

### Change 003

- Type: Modified file
- Path: `src/alpamayo1_5/test_inference.py`
- Reason:
  - Added top-level NVTX markers for the end-to-end inference driver so Nsight timelines show setup and runtime phases.
- Inserted markers:
  - `load_dataset`
  - `build_messages`
  - `load_model`
  - `build_processor`
  - `tokenize_inputs`
  - `move_to_device`
  - `full_inference`
- Rollback:
  - Revert the file to the previous revision in git:
    - `git -C /home/jys/alpamayo1.5 checkout -- /home/jys/alpamayo1.5/src/alpamayo1_5/test_inference.py`
