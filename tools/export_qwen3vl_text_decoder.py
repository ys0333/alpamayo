#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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
        description="Extract the text decoder from a Qwen3-VL HF checkpoint into a Qwen3 HF checkpoint."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_index(input_dir: Path) -> Dict:
    with open(input_dir / "model.safetensors.index.json", "r") as f:
        return json.load(f)


def iter_text_tensors(input_dir: Path, index: Dict) -> List[Tuple[str, str]]:
    weight_map = index["weight_map"]
    items = []
    for key, shard in weight_map.items():
        if key.startswith("model.language_model.") or key == "lm_head.weight":
            items.append((key, shard))
    return items


def target_name(source_name: str) -> str:
    if source_name.startswith("model.language_model."):
        return "model." + source_name[len("model.language_model."):]
    return source_name


def build_text_config(vlm_config: Dict) -> Dict:
    text_cfg = dict(vlm_config["text_config"])
    rope_scaling = text_cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        if rope_scaling.get("mrope_section") is not None or rope_scaling.get("mrope_interleaved"):
            rope_scaling["type"] = "mrope"
            rope_scaling["rope_type"] = "mrope"
        elif rope_scaling.get("rope_type") == "default":
            rope_scaling["type"] = "rope_gpt_neox"
            rope_scaling["rope_type"] = "rope_gpt_neox"
        text_cfg["rope_scaling"] = rope_scaling
    text_cfg["architectures"] = ["Qwen3ForCausalLM"]
    text_cfg["model_type"] = "qwen3"
    return text_cfg


def copy_tokenizer_files(input_dir: Path, output_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def repack_text_weights(input_dir: Path, output_dir: Path, index: Dict,
                        max_shard_size_bytes: int) -> Tuple[Dict[str, str], int, int]:
    items = iter_text_tensors(input_dir, index)
    current: Dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 1
    total_size = 0
    total_parameters = 0
    weight_map: Dict[str, str] = {}

    def flush() -> None:
        nonlocal current, current_size, shard_idx
        if not current:
            return
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(current, str(output_dir / shard_name))
        for key in current:
            weight_map[key] = shard_name
        current = {}
        current_size = 0
        shard_idx += 1

    cache: Dict[str, safe_open] = {}
    try:
        for source_name, source_shard in items:
            shard_path = input_dir / source_shard
            if source_shard not in cache:
                cache[source_shard] = safe_open(str(shard_path),
                                                framework="pt",
                                                device="cpu")
            tensor = cache[source_shard].get_tensor(source_name)
            new_name = target_name(source_name)
            tensor_size = tensor.numel() * tensor.element_size()
            if current and current_size + tensor_size > max_shard_size_bytes:
                flush()
            current[new_name] = tensor
            current_size += tensor_size
            total_size += tensor_size
            total_parameters += tensor.numel()
        flush()
    finally:
        cache.clear()

    shard_count = shard_idx - 1
    final_weight_map = {}
    for key, shard_name in weight_map.items():
        final_weight_map[key] = shard_name.replace("PLACEHOLDER",
                                                   f"{shard_count:05d}")
    for old_name in list({v for v in weight_map.values()}):
        new_name = old_name.replace("PLACEHOLDER", f"{shard_count:05d}")
        (output_dir / old_name).rename(output_dir / new_name)

    return final_weight_map, total_size, total_parameters


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output-dir already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(input_dir)
    with open(input_dir / "config.json", "r") as f:
        vlm_config = json.load(f)

    text_config = build_text_config(vlm_config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(text_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    copy_tokenizer_files(input_dir, output_dir)

    weight_map, total_size, total_parameters = repack_text_weights(
        input_dir=input_dir,
        output_dir=output_dir,
        index=index,
        max_shard_size_bytes=int(args.max_shard_size_gb * (1024**3)),
    )
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                    "total_parameters": total_parameters,
                },
                "weight_map": dict(sorted(weight_map.items())),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_tensors": len(weight_map),
                "total_size": total_size,
                "total_parameters": total_parameters,
            },
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
