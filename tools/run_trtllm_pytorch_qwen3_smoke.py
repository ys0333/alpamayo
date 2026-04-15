#!/usr/bin/env python3
"""Smoke test a Qwen3 HF checkpoint on TensorRT-LLM PyTorch backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test a Qwen3 HF checkpoint on TensorRT-LLM PyTorch backend."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Summarize the current scene.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-num-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-overlap-scheduler", action="store_true")
    parser.add_argument("--disable-flashinfer-sampling", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    if args.print_config:
        config = json.loads((args.model_dir / "config.json").read_text())
        print(
            json.dumps(
                {
                    "model_dir": str(args.model_dir),
                    "architectures": config.get("architectures"),
                    "model_type": config.get("model_type"),
                    "hidden_size": config.get("hidden_size"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(
        model=args.model_dir,
        backend="pytorch",
        trust_remote_code=args.trust_remote_code,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len,
        disable_overlap_scheduler=args.disable_overlap_scheduler,
        disable_flashinfer_sampling=args.disable_flashinfer_sampling,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    output = llm.generate(args.prompt, sampling_params=sampling_params, use_tqdm=False)
    payload = {
        "prompt": args.prompt,
        "output_text": output.outputs[0].text if output.outputs else None,
        "output_token_ids": output.outputs[0].token_ids if output.outputs else None,
        "finished": output.finished,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
