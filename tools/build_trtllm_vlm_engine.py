#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json", "r") as f:
        return json.load(f)


def infer_architecture(config: dict) -> str:
    architectures = config.get("architectures") or []
    if architectures:
        return architectures[0]
    return config.get("architecture") or config.get("model_type") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT-LLM engine from a local HF checkpoint supported by the current TensorRT backend."
    )
    parser.add_argument("--model-dir",
                        type=Path,
                        required=True,
                        help="Path to the repacked HF checkpoint directory.")
    parser.add_argument("--engine-dir",
                        type=Path,
                        required=True,
                        help="Output directory for the built TensorRT-LLM engine.")
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        help="Build dtype passed to TensorRT-LLM.")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--max-beam-width", type=int, default=1)
    parser.add_argument("--max-encoder-input-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--use-mrope", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unsupported-arch",
        action="store_true",
        help="Skip the local architecture guard. Use only when testing a patched TRT-LLM install.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    engine_dir = args.engine_dir.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")

    config = load_config(model_dir)
    architecture = infer_architecture(config)
    if architecture == "Qwen3VLForConditionalGeneration" and not args.allow_unsupported_arch:
        raise ValueError(
            "The current TensorRT backend API does not accept full "
            "Qwen3VLForConditionalGeneration checkpoints directly in this install. "
            "First extract the text decoder with "
            "`tools/export_qwen3vl_text_decoder.py`, then build the text engine from that output."
        )

    if engine_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"engine-dir already exists: {engine_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(engine_dir)

    build_config = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        max_beam_width=args.max_beam_width,
        max_encoder_input_len=args.max_encoder_input_len,
        dry_run=args.dry_run,
        use_mrope=args.use_mrope,
    )

    print("=== TRT-LLM VLM Build Config ===")
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "engine_dir": str(engine_dir),
                "architecture": architecture,
                "dtype": args.dtype,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpus_per_node": args.gpus_per_node,
                "build_config": build_config.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        ))

    start = time.time()
    llm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        skip_tokenizer_init=False,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        build_config=build_config,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len,
        max_beam_width=args.max_beam_width,
        gpus_per_node=args.gpus_per_node,
    )
    build_elapsed = time.time() - start
    print(f"Engine initialization/build finished in {build_elapsed:.2f}s")

    if args.dry_run:
        print("Dry-run requested; skipping engine save.")
        return

    save_start = time.time()
    llm.save(str(engine_dir))
    save_elapsed = time.time() - save_start
    print(f"Engine saved to {engine_dir} in {save_elapsed:.2f}s")


if __name__ == "__main__":
    main()
