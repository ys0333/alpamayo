#!/usr/bin/env python3
"""Export Alpamayo expert weights into a Qwen3 Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack Alpamayo expert weights into a Qwen3ForCausalLM-style checkpoint."
    )
    parser.add_argument("--alpamayo-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="HF directory that already contains the expanded Alpamayo tokenizer files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_expert_config(
    alpamayo_config: dict[str, Any],
    vlm_config: dict[str, Any],
) -> dict[str, Any]:
    text_config = dict(vlm_config["text_config"])
    rope_scaling = text_config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        if rope_scaling.get("mrope_section") is not None or rope_scaling.get("mrope_interleaved"):
            rope_scaling["type"] = "mrope"
            rope_scaling["rope_type"] = "mrope"
        elif rope_scaling.get("rope_type") == "default":
            rope_scaling["type"] = "rope_gpt_neox"
            rope_scaling["rope_type"] = "rope_gpt_neox"
        text_config["rope_scaling"] = rope_scaling
    for key, value in alpamayo_config.get("expert_cfg", {}).items():
        text_config[key] = value
    text_config["architectures"] = ["Qwen3ForCausalLM"]
    text_config["model_type"] = "qwen3"
    text_config["torch_dtype"] = alpamayo_config.get("model_dtype", "bfloat16")
    text_config["alpamayo_metadata"] = {
        "source_model_type": alpamayo_config["model_type"],
        "export_type": "expert_only",
        "expert_cfg": alpamayo_config.get("expert_cfg"),
    }
    return text_config


def collect_expert_groups(
    alpamayo_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    index_payload = load_json(alpamayo_dir / "model.safetensors.index.json")
    weight_map = index_payload["weight_map"]
    grouped: dict[str, list[str]] = defaultdict(list)

    for source_key, shard_name in weight_map.items():
        if source_key.startswith("expert."):
            grouped[shard_name].append(source_key)

    return grouped, weight_map


def target_key(source_key: str) -> str:
    if source_key.startswith("expert."):
        return "model." + source_key.removeprefix("expert.")
    raise KeyError(f"Unexpected source key: {source_key}")


def copy_tokenizer_files(tokenizer_dir: Path, output_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def repack_weights(
    alpamayo_dir: Path,
    output_dir: Path,
    grouped_keys: dict[str, list[str]],
    hidden_size: int,
    vocab_size: int,
    dtype_name: str,
) -> tuple[dict[str, str], int]:
    output_weight_map: dict[str, str] = {}
    total_parameters = 0
    total_shards = len(grouped_keys)
    dtype = getattr(torch, dtype_name, torch.bfloat16)

    for shard_idx, source_shard_name in enumerate(sorted(grouped_keys), start=1):
        tensors = {}
        output_shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        with safe_open(alpamayo_dir / source_shard_name, framework="pt", device="cpu") as source_file:
            for source_key in grouped_keys[source_shard_name]:
                tensor = source_file.get_tensor(source_key)
                tensors[target_key(source_key)] = tensor
                output_weight_map[target_key(source_key)] = output_shard_name
                total_parameters += tensor.numel()
        if shard_idx == 1:
            # Alpamayo expert consumes `inputs_embeds` and does not use token embeddings or lm_head.
            # We still provide shape-compatible tensors so TRT-LLM can instantiate a causal LM wrapper.
            embed = torch.zeros((vocab_size, hidden_size), dtype=dtype)
            lm_head = torch.zeros((vocab_size, hidden_size), dtype=dtype)
            tensors["model.embed_tokens.weight"] = embed
            tensors["lm_head.weight"] = lm_head
            output_weight_map["model.embed_tokens.weight"] = output_shard_name
            output_weight_map["lm_head.weight"] = output_shard_name
            total_parameters += embed.numel() + lm_head.numel()
        save_file(tensors, output_dir / output_shard_name, metadata={"format": "pt"})

    return output_weight_map, total_parameters


def main() -> None:
    args = parse_args()
    alpamayo_dir = args.alpamayo_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    output_dir = args.output_dir.resolve()

    alpamayo_config = load_json(alpamayo_dir / "config.json")
    vlm_config = load_json(tokenizer_dir / "config.json")
    grouped_keys, weight_map = collect_expert_groups(alpamayo_dir)
    export_config = build_expert_config(alpamayo_config, vlm_config)
    hidden_size = int(export_config["hidden_size"])
    vocab_size = int(export_config["vocab_size"])
    dtype_name = str(export_config.get("torch_dtype", "bfloat16"))

    summary = {
        "alpamayo_dir": str(alpamayo_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "output_dir": str(output_dir),
        "expert_tensor_count": sum(len(keys) for keys in grouped_keys.values()),
        "expert_shard_count": len(grouped_keys),
        "dummy_embed_shape": [vocab_size, hidden_size],
        "dummy_lm_head_shape": [vocab_size, hidden_size],
        "sample_target_keys": [
            target_key(key)
            for key in (
                "expert.layers.0.self_attn.q_proj.weight",
            )
            if key in weight_map
        ]
        + ["model.embed_tokens.weight", "lm_head.weight"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    ensure_output_dir(output_dir, overwrite=args.overwrite)
    save_json(output_dir / "config.json", export_config)
    copy_tokenizer_files(tokenizer_dir, output_dir)
    output_weight_map, total_parameters = repack_weights(
        alpamayo_dir,
        output_dir,
        grouped_keys,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype_name=dtype_name,
    )
    save_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {
                "total_parameters": total_parameters,
                "total_size": sum(
                    (output_dir / shard_name).stat().st_size
                    for shard_name in sorted(set(output_weight_map.values()))
                ),
            },
            "weight_map": dict(sorted(output_weight_map.items())),
        },
    )
    print(f"Wrote repacked expert checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
