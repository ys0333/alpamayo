"""TensorRT-LLM PyTorch backend adapters for Alpamayo submodules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from alpamayo1_5.models.token_utils import extract_text_tokens


def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor frame into a PIL image for TRT-LLM multimodal input."""
    frame = frame.detach().cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (C, H, W), got {tuple(frame.shape)}")

    if frame.dtype != torch.uint8:
        if frame.is_floating_point():
            frame = frame.clamp(0, 255).to(torch.uint8)
        else:
            frame = frame.to(torch.uint8)

    frame = frame.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(frame)


def _extract_mm_images(data: dict[str, Any], batch_size: int) -> list[list[Image.Image] | None]:
    image_frames = data.get("image_frames")
    if image_frames is None:
        return [None] * batch_size

    if not isinstance(image_frames, torch.Tensor):
        raise TypeError("Expected data['image_frames'] to be a torch.Tensor when present.")

    if image_frames.ndim == 4:
        images = [_frame_to_pil(frame) for frame in image_frames]
        return [images for _ in range(batch_size)]

    if image_frames.ndim == 5:
        if image_frames.shape[0] != batch_size:
            raise ValueError(
                "Batched image_frames first dimension must match prompt batch size: "
                f"{image_frames.shape[0]} != {batch_size}"
            )
        return [[_frame_to_pil(frame) for frame in sample] for sample in image_frames]

    raise ValueError(
        "Expected image_frames to have shape (N, C, H, W) or (B, N, C, H, W), "
        f"got {tuple(image_frames.shape)}"
    )


class TrtllmQwen3VlmPytorchBackend:
    """Lazy wrapper around TensorRT-LLM PyTorch backend for Alpamayo VLM generation."""

    def __init__(
        self,
        *,
        model_dir: str,
        trust_remote_code: bool,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        disable_overlap_scheduler: bool,
        disable_flashinfer_sampling: bool,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.trust_remote_code = trust_remote_code
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.disable_flashinfer_sampling = disable_flashinfer_sampling
        self._llm = None

    def _ensure_llm(self):
        if self._llm is None:
            from tensorrt_llm import LLM

            self._llm = LLM(
                model=self.model_dir,
                backend="pytorch",
                trust_remote_code=self.trust_remote_code,
                max_batch_size=self.max_batch_size,
                max_num_tokens=self.max_num_tokens,
                max_seq_len=self.max_seq_len,
                disable_overlap_scheduler=self.disable_overlap_scheduler,
                disable_flashinfer_sampling=self.disable_flashinfer_sampling,
            )
        return self._llm

    def shutdown(self) -> None:
        if self._llm is not None:
            self._llm.shutdown()
            self._llm = None

    def _build_requests(self, data: dict[str, Any], input_ids: torch.Tensor) -> list[dict[str, Any]]:
        batch_size = input_ids.shape[0]
        mm_images = _extract_mm_images(data, batch_size)
        requests = []
        for batch_idx in range(batch_size):
            request: dict[str, Any] = {
                "prompt_token_ids": input_ids[batch_idx].detach().cpu().tolist(),
            }
            if mm_images[batch_idx] is not None:
                request["multi_modal_data"] = {"image": mm_images[batch_idx]}
            requests.append(request)
        return requests

    def generate_text(
        self,
        *,
        data: dict[str, Any],
        input_ids: torch.Tensor,
        tokenizer: Any,
        top_p: float,
        top_k: int | None,
        temperature: float,
        num_samples: int,
        max_generation_length: int,
    ) -> dict[str, np.ndarray]:
        from tensorrt_llm import SamplingParams

        llm = self._ensure_llm()
        requests = self._build_requests(data, input_ids)
        sampling_params = SamplingParams(
            max_tokens=max_generation_length,
            n=num_samples,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_id=tokenizer.pad_token_id,
            exclude_input_from_output=True,
        )
        outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        if not isinstance(outputs, list):
            outputs = [outputs]

        flat_token_rows: list[list[int]] = []
        for request_output in outputs:
            if not request_output.outputs:
                flat_token_rows.extend([[] for _ in range(num_samples)])
                continue
            for completion in request_output.outputs[:num_samples]:
                flat_token_rows.append(list(completion.token_ids))

        if len(flat_token_rows) != input_ids.shape[0] * num_samples:
            raise RuntimeError(
                "Unexpected number of TRT-LLM outputs: "
                f"{len(flat_token_rows)} != {input_ids.shape[0] * num_samples}"
            )

        max_len = max((len(row) for row in flat_token_rows), default=0)
        if max_len == 0:
            generated_tokens = torch.empty(
                (len(flat_token_rows), 0), dtype=torch.long, device=input_ids.device
            )
        else:
            generated_tokens = torch.full(
                (len(flat_token_rows), max_len),
                fill_value=tokenizer.pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            for row_idx, token_row in enumerate(flat_token_rows):
                if token_row:
                    generated_tokens[row_idx, : len(token_row)] = torch.tensor(
                        token_row, dtype=torch.long, device=input_ids.device
                    )

        extra = extract_text_tokens(tokenizer, generated_tokens)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([input_ids.shape[0], num_samples])
        return extra
