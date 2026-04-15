#!/usr/bin/env python3
"""Repack Alpamayo VLM weights into a Qwen3-VL Hugging Face checkpoint.

This script extracts only the `vlm.*` portion from an Alpamayo 1.5 checkpoint,
reconstructs the tokenizer/processor with Alpamayo's extra trajectory tokens,
and writes a standalone Qwen3-VL-style Hugging Face checkpoint directory that
TensorRT-LLM can consume through its Qwen3-VL path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {k: f"<|{k}|>" for k in SPECIAL_TOKENS_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpamayo-dir", type=Path, required=True)
    parser.add_argument("--base-vlm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to continue.")
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def reconstruct_processor(
    base_vlm_dir: Path,
    output_dir: Path,
    alpamayo_config: dict[str, Any],
) -> None:
    processor = AutoProcessor.from_pretrained(base_vlm_dir, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = alpamayo_config.get("padding_side", tokenizer.padding_side)

    traj_vocab_size = int(alpamayo_config["traj_vocab_size"])
    tokenizer.add_tokens([f"<i{idx}>" for idx in range(traj_vocab_size)])

    if alpamayo_config.get("add_special_tokens", False):
        extra_tokens = list(SPECIAL_TOKENS.values())
    else:
        extra_tokens = list(TRAJ_TOKEN.values())
    tokenizer.add_tokens(extra_tokens, special_tokens=True)

    processor.tokenizer = tokenizer
    processor.save_pretrained(output_dir)

    traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    if traj_token_start_idx != int(alpamayo_config["traj_token_start_idx"]):
        raise ValueError(
            "Tokenizer reconstruction mismatch for traj_token_start_idx: "
            f"{traj_token_start_idx} != {alpamayo_config['traj_token_start_idx']}"
        )

    for key, expected_token_id in alpamayo_config["traj_token_ids"].items():
        token = TRAJ_TOKEN[key]
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != int(expected_token_id):
            raise ValueError(
                f"Tokenizer reconstruction mismatch for {key}: {actual} != {expected_token_id}"
            )

    if len(tokenizer) != int(alpamayo_config["vocab_size"]):
        raise ValueError(
            f"Tokenizer size mismatch: {len(tokenizer)} != {alpamayo_config['vocab_size']}"
        )


def build_output_config(
    base_config: dict[str, Any],
    alpamayo_config: dict[str, Any],
) -> dict[str, Any]:
    output_config = json.loads(json.dumps(base_config))
    output_config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    output_config["model_type"] = "qwen3_vl"
    output_config["text_config"]["vocab_size"] = int(alpamayo_config["vocab_size"])
    output_config["text_config"]["dtype"] = alpamayo_config.get(
        "model_dtype", output_config["text_config"].get("dtype", "bfloat16")
    )
    output_config["torch_dtype"] = alpamayo_config.get("model_dtype", "bfloat16")
    output_config["alpamayo_metadata"] = {
        "source_model_type": alpamayo_config["model_type"],
        "traj_vocab_size": alpamayo_config["traj_vocab_size"],
        "traj_token_start_idx": alpamayo_config["traj_token_start_idx"],
        "traj_token_ids": alpamayo_config["traj_token_ids"],
        "tokens_per_history_traj": alpamayo_config["tokens_per_history_traj"],
        "tokens_per_future_traj": alpamayo_config["tokens_per_future_traj"],
        "padding_side": alpamayo_config.get("padding_side", "left"),
        "base_vlm_name_or_path": alpamayo_config["vlm_name_or_path"],
    }
    return output_config


def collect_vlm_weight_map(alpamayo_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    index_payload = load_json(alpamayo_dir / "model.safetensors.index.json")
    weight_map = index_payload["weight_map"]
    vlm_weight_map: dict[str, str] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_key, shard_name in weight_map.items():
        if not source_key.startswith("vlm."):
            continue
        target_key = source_key.removeprefix("vlm.")
        vlm_weight_map[target_key] = shard_name
        grouped[shard_name].append(source_key)
    return vlm_weight_map, grouped


def repack_vlm_weights(
    alpamayo_dir: Path,
    output_dir: Path,
    grouped_source_keys: dict[str, list[str]],
) -> tuple[dict[str, str], int]:
    total_shards = len(grouped_source_keys)
    output_weight_map: dict[str, str] = {}
    total_parameters = 0

    for shard_idx, source_shard_name in enumerate(sorted(grouped_source_keys), start=1):
        source_path = alpamayo_dir / source_shard_name
        output_shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        output_path = output_dir / output_shard_name
        print(f"Repacking {source_shard_name} -> {output_shard_name}")

        shard_tensors = {}
        with safe_open(source_path, framework="pt", device="cpu") as source_file:
            for source_key in grouped_source_keys[source_shard_name]:
                target_key = source_key.removeprefix("vlm.")
                tensor = source_file.get_tensor(source_key)
                total_parameters += tensor.numel()
                shard_tensors[target_key] = tensor
                output_weight_map[target_key] = output_shard_name

        save_file(shard_tensors, output_path, metadata={"format": "pt"})

    return output_weight_map, total_parameters


def build_index_payload(
    output_dir: Path,
    output_weight_map: dict[str, str],
    total_parameters: int,
) -> dict[str, Any]:
    total_size = 0
    for shard_name in sorted(set(output_weight_map.values())):
        total_size += (output_dir / shard_name).stat().st_size
    return {
        "metadata": {
            "total_parameters": total_parameters,
            "total_size": total_size,
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }


def print_summary(
    base_config: dict[str, Any],
    alpamayo_config: dict[str, Any],
    grouped_source_keys: dict[str, list[str]],
) -> None:
    print("Base architecture:", base_config["architectures"][0])
    print("Base vocab size:", base_config["text_config"]["vocab_size"])
    print("Alpamayo vocab size:", alpamayo_config["vocab_size"])
    print("Alpamayo traj token start:", alpamayo_config["traj_token_start_idx"])
    print("Output shard count:", len(grouped_source_keys))
    for shard_name in sorted(grouped_source_keys):
        print(f"  {shard_name}: {len(grouped_source_keys[shard_name])} VLM tensors")


def main() -> None:
    args = parse_args()
    alpamayo_dir = args.alpamayo_dir.resolve()
    base_vlm_dir = args.base_vlm_dir.resolve()
    output_dir = args.output_dir.resolve()

    alpamayo_config = load_json(alpamayo_dir / "config.json")
    base_config = load_json(base_vlm_dir / "config.json")
    output_config = build_output_config(base_config, alpamayo_config)
    _, grouped_source_keys = collect_vlm_weight_map(alpamayo_dir)

    print_summary(base_config, alpamayo_config, grouped_source_keys)
    if args.dry_run:
        print("Dry-run only. No files written.")
        return

    ensure_output_dir(output_dir, overwrite=args.overwrite)
    reconstruct_processor(base_vlm_dir, output_dir, alpamayo_config)
    save_json(output_dir / "config.json", output_config)

    output_weight_map, total_parameters = repack_vlm_weights(
        alpamayo_dir, output_dir, grouped_source_keys
    )
    save_json(
        output_dir / "model.safetensors.index.json",
        build_index_payload(output_dir, output_weight_map, total_parameters),
    )

    print(f"Wrote repacked checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
