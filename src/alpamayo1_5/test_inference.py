# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example script for the inference pipeline.

Loads a dataset, runs inference, and computes the minADE.
"""

import argparse
import time

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import (
    Alpamayo1_5,
    ExpertLogitsProcessor,
    nvtx_range,
)
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)


def _parse_cross_attention_layers(raw_value: str | None) -> str | list[int] | None:
    """Parse a CLI cross-attention layer spec."""
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    if value.lower() == "all":
        return "all"
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Alpamayo 1.5 inference on an example clip.")
    parser.add_argument(
        "--clip-id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
        help="Clip ID to evaluate.",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=5_100_000,
        help="Starting timestamp in microseconds for the first inference sample.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=1,
        help="Number of consecutive inference windows to run on the same clip.",
    )
    parser.add_argument(
        "--t0-step-us",
        type=int,
        default=100_000,
        help="Microsecond step between consecutive inference windows. Default 100000 = 0.1s.",
    )
    parser.add_argument(
        "--expert-cross-attention-layers",
        type=str,
        default=None,
        help='Enable expert cross-attention on selected expert blocks. Use "all" or comma-separated layer indices such as "24,28,32,35".',
    )
    parser.add_argument(
        "--select-prefix",
        type=str,
        default=None,
        help='Keep prefix self-attention only on selected expert blocks. Use "all" or comma-separated layer indices such as "24,28,32,35". When set, unselected layers ignore prefix KV cache.',
    )
    parser.add_argument(
        "--keep-prefix-self-attention",
        action="store_true",
        help="Keep the original prefix-based self-attention in cross-attention-enabled expert blocks. By default, selected blocks use cross-attention as the only conditioning path.",
    )
    parser.add_argument(
        "--vlm-only",
        action="store_true",
        help="Run only the VLM generate stage and skip diffusion/planning.",
    )
    parser.add_argument(
        "--print-block",
        action="store_true",
        help="Print per-block attention workload stats for the first expert forward in each inference.",
    )
    parser.add_argument(
        "--ablate-head",
        type=str,
        default=None,
        help="Ablate one attention head using format '<layer>:<head>' (0-based).",
    )
    parser.add_argument(
        "--ablate-dim",
        type=str,
        default=None,
        help="Ablate one attention sub-dimension using format '<layer>:<head>:<dim>' (all 0-based).",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference on an example clip and report minADE."""
    args = parse_args()
    expert_cross_attention_layers = _parse_cross_attention_layers(
        args.expert_cross_attention_layers
    )
    expert_prefix_layers = _parse_cross_attention_layers(args.select_prefix)

    with nvtx_range("load_model"):
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            expert_cross_attention_layers=expert_cross_attention_layers,
            expert_cross_attention_replace_prefix=not args.keep_prefix_self_attention,
            expert_prefix_layers=expert_prefix_layers,
            expert_ablate_head=args.ablate_head,
            expert_ablate_dim=args.ablate_dim,
        ).to("cuda")
    with nvtx_range("build_processor"):
        processor = helper.get_processor(model.tokenizer)

    torch.cuda.manual_seed_all(42)
    min_ades = []
    inference_latencies_ms = []
    diffusion_latencies_ms = []

    for sample_idx in range(args.nums):
        t0_us = args.t0_us + sample_idx * args.t0_step_us
        print(
            f"[{sample_idx + 1}/{args.nums}] "
            f"Loading dataset for clip_id: {args.clip_id}, t0_us: {t0_us}..."
        )
        with nvtx_range("load_dataset"):
            data = load_physical_aiavdataset(args.clip_id, t0_us=t0_us)
        print("Dataset loaded.")
        with nvtx_range("build_messages"):
            messages = helper.create_message(
                frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
            )

        with nvtx_range("tokenize_inputs"):
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }

        with nvtx_range("move_to_device"):
            model_inputs = helper.to_device(model_inputs, "cuda")

        inference_start = time.perf_counter()
        if args.vlm_only:
            with nvtx_range("vlm_only_inference"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    extra = run_vlm_only_inference(model, model_inputs)
            inference_latency_ms = (time.perf_counter() - inference_start) * 1000.0
            print("Chain-of-Causation (per trajectory):\n", np.array(extra["cot"]).reshape(1, 1))
            print(f"vlm-only latency: {inference_latency_ms:.2f} ms")
            continue

        with nvtx_range("full_inference"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                    return_timings=True,
                    return_attention_debug=args.print_block,
                )
        inference_latency_ms = (time.perf_counter() - inference_start) * 1000.0
        timings_ms = extra.get("timings_ms", {})
        diffusion_latency_ms = timings_ms.get("diffusion_sample")

        print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        min_ades.append(float(min_ade))
        inference_latencies_ms.append(float(inference_latency_ms))
        if diffusion_latency_ms is not None:
            diffusion_latencies_ms.append(float(diffusion_latency_ms))

        print("minADE:", min_ade, "meters")
        print(f"inference latency: {inference_latency_ms:.2f} ms")
        if diffusion_latency_ms is not None:
            print(f"diffusion latency: {diffusion_latency_ms:.2f} ms")
            print(
                f"diffusion / e2e ratio: {100.0 * diffusion_latency_ms / inference_latency_ms:.2f}%"
            )
        if args.print_block and "attention_debug" in extra:
            _print_attention_debug(extra["attention_debug"])
        if min_ade >= 1.0:
            print(
                f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic."
            )

    if args.nums > 1:
        summary = (
            f"Summary over {args.nums} runs: "
            f"mean minADE={np.mean(min_ades):.4f}m, "
            f"mean e2e latency={np.mean(inference_latencies_ms):.2f} ms"
        )
        if diffusion_latencies_ms:
            summary += (
                f", mean diffusion latency={np.mean(diffusion_latencies_ms):.2f} ms"
                f", mean diffusion / e2e ratio="
                f"{100.0 * np.mean(diffusion_latencies_ms) / np.mean(inference_latencies_ms):.2f}%"
            )
        print(summary)


def run_vlm_only_inference(
    model: Alpamayo1_5,
    model_inputs: dict[str, torch.Tensor],
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    max_generation_length: int = 256,
) -> dict[str, list[str]]:
    """Run only the VLM generation stage used by Alpamayo inference."""
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }

    with nvtx_range("fuse_traj_tokens"):
        input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    with nvtx_range("vlm_generate"):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )

    with nvtx_range("postprocess_generate_outputs"):
        sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
        )
        return extract_text_tokens(model.tokenizer, sequences)


def _print_attention_debug(attention_debug: dict[str, object]) -> None:
    """Pretty-print per-layer attention workload estimates."""
    expert_forward_calls = attention_debug.get("expert_forward_calls", 0)
    self_layers = attention_debug.get("self_attn_layers", [])
    cross_layers = attention_debug.get("cross_attn_layers", [])
    print(f"expert forward calls: {expert_forward_calls}")
    print("self-attention workload by layer:")
    total_self_scores = 0
    for layer in self_layers:
        total_self_scores += int(layer["attention_scores"])
        prefix_tag = "prefix" if layer["uses_prefix"] else "local"
        print(
            "  "
            f"layer={layer['layer_idx']:>2} mode={prefix_tag:<6} "
            f"q={layer['query_length']:>3} kv={layer['kv_length']:>4} "
            f"heads={layer['num_heads']:>2} scores={layer['attention_scores']}"
        )
    print(f"estimated self-attention scores per expert forward: {total_self_scores}")
    if cross_layers:
        print("cross-attention workload by layer:")
        total_cross_scores = 0
        for layer in cross_layers:
            total_cross_scores += int(layer["attention_scores"])
            print(
                "  "
                f"layer={layer['layer_idx']:>2} q={layer['query_length']:>3} "
                f"kv={layer['kv_length']:>4} heads={layer['num_heads']:>2} "
                f"scores={layer['attention_scores']}"
            )
        print(f"estimated cross-attention scores per expert forward: {total_cross_scores}")


if __name__ == "__main__":
    main()
