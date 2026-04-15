#!/usr/bin/env python3
"""Direct TRT-LLM _torch forward smoke test for Alpamayo expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a direct TensorRT-LLM _torch forward pass on the exported Alpamayo expert."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--attn-backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM"],
    )
    parser.add_argument(
        "--attention-mask",
        type=str,
        default="full",
        choices=["full", "causal"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM

    model_config = ModelConfig.from_pretrained(
        str(args.model_dir),
        attn_backend=args.attn_backend,
        max_num_tokens=args.num_tokens * args.batch_size,
        max_seq_len=args.num_tokens,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_config(model_config).to("cuda")

    loader = HfCheckpointLoader()
    weights = loader.load_weights(str(args.model_dir), mapping=model_config.mapping)
    weight_mapper = loader.get_initialized_weight_mapper(model, model_config)
    model.load_weights(weights, weight_mapper)
    loader.cleanup()

    config = json.loads((args.model_dir / "config.json").read_text())
    hidden_size = int(config["hidden_size"])
    total_tokens = args.batch_size * args.num_tokens
    inputs_embeds = torch.randn(
        total_tokens,
        hidden_size,
        device="cuda",
        dtype=model_config.torch_dtype,
    )
    position_ids = (
        torch.arange(args.num_tokens, device="cuda", dtype=torch.int32)
        .view(1, 1, args.num_tokens)
        .expand(3, args.batch_size, args.num_tokens)
        .contiguous()
    )
    seq_lens = torch.tensor([args.num_tokens] * args.batch_size, dtype=torch.int32)

    metadata_cls = (
        VanillaAttentionMetadata if args.attn_backend == "VANILLA" else TrtllmAttentionMetadata
    )
    attn_metadata = metadata_cls(
        max_num_requests=args.batch_size,
        max_num_tokens=total_tokens,
        max_num_sequences=args.batch_size,
        kv_cache_manager=None,
        mapping=model_config.mapping,
        request_ids=list(range(args.batch_size)),
        seq_lens=seq_lens,
        # Leave KV sequence lengths unset for no-cache self-attention.
        # In TRT-LLM _torch, setting a distinct tensor here marks the batch as cross-attention.
        seq_lens_kv=None,
        prompt_lens=[args.num_tokens] * args.batch_size,
    )
    attn_metadata.num_contexts = args.batch_size
    attn_metadata.position_ids = position_ids
    attn_metadata.max_seq_len = args.num_tokens
    attn_metadata.prepare()

    attn_mask = (
        PredefinedAttentionMask.FULL
        if args.attention_mask == "full"
        else PredefinedAttentionMask.CAUSAL
    )

    with torch.inference_mode():
        outputs = model.model(
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn_mask,
        )

    payload = {
        "model_dir": str(args.model_dir),
        "attn_backend": args.attn_backend,
        "attention_mask": args.attention_mask,
        "batch_size": args.batch_size,
        "num_tokens": args.num_tokens,
        "hidden_size": hidden_size,
        "output_shape": list(outputs.shape),
        "output_dtype": str(outputs.dtype),
        "output_abs_mean": float(outputs.abs().mean().item()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
