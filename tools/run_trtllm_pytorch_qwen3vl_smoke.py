#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test a repacked Qwen3-VL checkpoint on TensorRT-LLM PyTorch backend."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the local HF-style Qwen3-VL checkpoint directory.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path for a multimodal generation smoke test.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image.",
        help="Prompt text to use for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for the TRT-LLM PyTorch backend.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens for the TRT-LLM PyTorch backend.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the TRT-LLM PyTorch backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to TensorRT-LLM.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the model config summary before loading.",
    )
    parser.add_argument(
        "--disable-overlap-scheduler",
        action="store_true",
        help="Disable TensorRT-LLM PyTorch backend overlap scheduler.",
    )
    parser.add_argument(
        "--disable-flashinfer-sampling",
        action="store_true",
        help="Disable FlashInfer sampling and force the fallback sampler.",
    )
    return parser


def load_prompt_inputs(prompt: str, image_path: Path | None):
    request = {"prompt": prompt}
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        request["multi_modal_data"] = {"image": [image]}
    return request


def main() -> int:
    args = build_parser().parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    if args.print_config:
        config = json.loads((args.model_dir / "config.json").read_text())
        print(
            json.dumps(
                {
                    "model_dir": str(args.model_dir),
                    "architectures": config.get("architectures"),
                    "model_type": config.get("model_type"),
                    "vocab_size": config.get("vocab_size"),
                },
                indent=2,
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

    prompt_inputs = load_prompt_inputs(args.prompt, args.image)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    result = llm.generate(
        prompt_inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    payload = {
        "prompt": args.prompt,
        "image": str(args.image) if args.image is not None else None,
        "output_text": result.outputs[0].text if result.outputs else None,
        "output_token_ids": result.outputs[0].token_ids if result.outputs else None,
        "finished": result.finished,
        "disable_overlap_scheduler": args.disable_overlap_scheduler,
        "disable_flashinfer_sampling": args.disable_flashinfer_sampling,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
