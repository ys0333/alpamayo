#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "tools" / "export_qwen3vl_text_decoder.py"
BUILD_SCRIPT = REPO_ROOT / "tools" / "build_trtllm_vlm_engine.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Qwen3 text decoder from a Qwen3-VL checkpoint and build a TensorRT-LLM text engine."
    )
    parser.add_argument("--vlm-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--max-beam-width", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    export_cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--input-dir",
        str(args.vlm_dir.resolve()),
        "--output-dir",
        str(args.text_dir.resolve()),
    ]
    if args.overwrite:
        export_cmd.append("--overwrite")
    run_step(export_cmd)

    build_cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--model-dir",
        str(args.text_dir.resolve()),
        "--engine-dir",
        str(args.engine_dir.resolve()),
        "--dtype",
        args.dtype,
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-input-len",
        str(args.max_input_len),
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-num-tokens",
        str(args.max_num_tokens),
        "--max-beam-width",
        str(args.max_beam_width),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpus-per-node",
        str(args.gpus_per_node),
    ]
    if args.trust_remote_code:
        build_cmd.append("--trust-remote-code")
    if args.overwrite:
        build_cmd.append("--overwrite")
    if args.dry_run:
        build_cmd.append("--dry-run")
    run_step(build_cmd)


if __name__ == "__main__":
    main()
