from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.cache_utils import Cache, DynamicCache

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token


@contextmanager
def nvtx_range(message: str):
    """Create an NVTX range when CUDA NVTX is available."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _maybe_to_legacy_cache(cache: Cache | tuple[Any, ...] | None) -> Cache | tuple[Any, ...] | None:
    if isinstance(cache, DynamicCache):
        return cache.to_legacy_cache()
    return cache


def _maybe_from_legacy_cache(cache: Cache | tuple[Any, ...] | None) -> Cache | None:
    if cache is None or isinstance(cache, Cache):
        return cache
    return DynamicCache.from_legacy_cache(cache)


@dataclass
class VlmGenerateArtifacts:
    sequences: torch.Tensor
    prompt_cache: Cache
    rope_deltas: torch.Tensor
    prefill_seq_len: int
    offset: torch.Tensor
    prefix_mask: torch.Tensor | None


class VisionEncoderWrapper(torch.nn.Module):
    """Wrap Qwen3-VL image feature extraction for profiling/export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.vlm = model.vlm

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        with nvtx_range("vision_encoder"):
            return self.vlm.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )


class VlmPrefillWrapper(torch.nn.Module):
    """Wrap a single VLM prefill forward pass."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.vlm = model.vlm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        logits_to_keep: int = 1,
        return_legacy_cache: bool = False,
    ) -> tuple[torch.Tensor, Cache | tuple[Any, ...], torch.Tensor | None]:
        with nvtx_range("vlm_prefill"):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
        cache = outputs.past_key_values
        if return_legacy_cache:
            cache = _maybe_to_legacy_cache(cache)
        return outputs.logits, cache, outputs.rope_deltas


class VlmDecodeStepWrapper(torch.nn.Module):
    """Wrap one autoregressive decode step with an existing KV cache."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.vlm = model.vlm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | tuple[Any, ...] | None,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int = 1,
        return_legacy_cache: bool = False,
    ) -> tuple[torch.Tensor, Cache | tuple[Any, ...], torch.Tensor | None]:
        cache = _maybe_from_legacy_cache(past_key_values)
        with nvtx_range("vlm_decode_step"):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
        cache = outputs.past_key_values
        if return_legacy_cache:
            cache = _maybe_to_legacy_cache(cache)
        return outputs.logits, cache, outputs.rope_deltas


class ActionInProjWrapper(torch.nn.Module):
    """Wrap action_in_proj for profiling/export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.action_in_proj = model.action_in_proj

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with nvtx_range("action_in_proj"):
            return self.action_in_proj(x, t)


class ExpertForwardWrapper(torch.nn.Module):
    """Wrap expert forward on already-projected future token embeddings."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.expert = model.expert

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | tuple[Any, ...],
        is_causal: bool = False,
        return_legacy_cache: bool = False,
    ) -> tuple[torch.Tensor, Cache | tuple[Any, ...]]:
        cache = _maybe_from_legacy_cache(past_key_values)
        with nvtx_range("expert_forward"):
            outputs = self.expert(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                is_causal=is_causal,
            )
        cache = outputs.past_key_values
        if return_legacy_cache:
            cache = _maybe_to_legacy_cache(cache)
        return outputs.last_hidden_state, cache


class ActionOutProjWrapper(torch.nn.Module):
    """Wrap action_out_proj for profiling/export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.action_out_proj = model.action_out_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with nvtx_range("action_out_proj"):
            return self.action_out_proj(hidden_states)


class ExpertDenoiserStepWrapper(torch.nn.Module):
    """Wrap action_in_proj -> expert -> action_out_proj as one denoiser step."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | tuple[Any, ...],
        return_legacy_cache: bool = False,
    ) -> tuple[torch.Tensor, Cache | tuple[Any, ...]]:
        cache = _maybe_from_legacy_cache(past_key_values)
        n_diffusion_tokens = self.model.action_space.get_action_space_dims()[0]
        with nvtx_range("expert_denoiser_step"):
            future_token_embeds = self.model.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)

            prefill_seq_len = cache.get_seq_length()
            expert_out = self.model.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                is_causal=not self.model.config.expert_non_causal_attention,
            )
            cache.crop(prefill_seq_len)
            hidden_states = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            pred = self.model.action_out_proj(hidden_states).view(
                -1, *self.model.action_space.get_action_space_dims()
            )

        out_cache: Cache | tuple[Any, ...] = cache
        if return_legacy_cache:
            out_cache = _maybe_to_legacy_cache(cache)
        return pred, out_cache


class DiffusionSampleWrapper(torch.nn.Module):
    """Wrap diffusion sampling with NVTX annotations around the outer loop."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.denoiser = ExpertDenoiserStepWrapper(model)

    def forward(
        self,
        batch_size: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | tuple[Any, ...],
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            pred, _ = self.denoiser(
                x=x,
                t=t,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            return pred

        with nvtx_range("diffusion_sample"):
            return self.model.diffusion.sample(
                batch_size=batch_size,
                step_fn=step_fn,
                device=position_ids.device,
                return_all_steps=False,
                **diffusion_kwargs,
            )


class AlpamayoProfilingWrappers:
    """Convenience accessors for component-level wrappers."""

    def __init__(self, model: torch.nn.Module):
        self.vision_encoder = VisionEncoderWrapper(model)
        self.vlm_prefill = VlmPrefillWrapper(model)
        self.vlm_decode_step = VlmDecodeStepWrapper(model)
        self.action_in_proj = ActionInProjWrapper(model)
        self.expert_forward = ExpertForwardWrapper(model)
        self.action_out_proj = ActionOutProjWrapper(model)
        self.expert_denoiser_step = ExpertDenoiserStepWrapper(model)
        self.diffusion_sample = DiffusionSampleWrapper(model)


def build_vlm_generate_artifacts(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenized_data: dict[str, Any],
    top_p: float,
    top_k: int | None,
    temperature: float,
    num_traj_samples: int,
    num_traj_sets: int,
    max_generation_length: int,
) -> VlmGenerateArtifacts:
    """Run the VLM rollout once and return tensors reused by downstream wrappers."""
    n_samples_total = num_traj_samples * num_traj_sets
    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
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

    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    offset = model._find_eos_offset(
        sequences=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        device=input_ids.device,
    )
    prefix_mask = tokenized_data.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)

    return VlmGenerateArtifacts(
        sequences=vlm_outputs.sequences,
        prompt_cache=prompt_cache,
        rope_deltas=vlm_outputs.rope_deltas,
        prefill_seq_len=prefill_seq_len,
        offset=offset,
        prefix_mask=prefix_mask,
    )


def build_diffusion_inputs_from_generate(
    model: torch.nn.Module,
    artifacts: VlmGenerateArtifacts,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build expert position ids and attention mask from cached VLM rollout state."""
    b_star = artifacts.sequences.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    return model._build_expert_pos_ids_and_attn_mask(
        offset=artifacts.offset,
        rope_deltas=artifacts.rope_deltas,
        kv_cache_seq_len=artifacts.prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=artifacts.sequences.device,
        prefix_mask=artifacts.prefix_mask,
    )
