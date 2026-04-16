from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_ATTENTION_DEBUG_STATE: dict[str, Any] = {
    "enabled": False,
    "capture_once": True,
    "captured": False,
    "self_attn_layers": [],
    "cross_attn_layers": [],
    "expert_forward_calls": 0,
}


def reset_attention_debug_state(enabled: bool = False, capture_once: bool = True) -> None:
    """Reset per-inference attention debug capture."""
    _ATTENTION_DEBUG_STATE["enabled"] = enabled
    _ATTENTION_DEBUG_STATE["capture_once"] = capture_once
    _ATTENTION_DEBUG_STATE["captured"] = False
    _ATTENTION_DEBUG_STATE["self_attn_layers"] = []
    _ATTENTION_DEBUG_STATE["cross_attn_layers"] = []
    _ATTENTION_DEBUG_STATE["expert_forward_calls"] = 0


def get_attention_debug_state() -> dict[str, Any]:
    """Return a copy of the current attention debug state."""
    return {
        "enabled": _ATTENTION_DEBUG_STATE["enabled"],
        "capture_once": _ATTENTION_DEBUG_STATE["capture_once"],
        "captured": _ATTENTION_DEBUG_STATE["captured"],
        "expert_forward_calls": _ATTENTION_DEBUG_STATE["expert_forward_calls"],
        "self_attn_layers": list(_ATTENTION_DEBUG_STATE["self_attn_layers"]),
        "cross_attn_layers": list(_ATTENTION_DEBUG_STATE["cross_attn_layers"]),
    }


def begin_attention_debug_capture() -> None:
    """Mark the start of one expert forward call for debug capture."""
    if not _ATTENTION_DEBUG_STATE["enabled"]:
        return
    _ATTENTION_DEBUG_STATE["expert_forward_calls"] += 1


def _should_capture_attention_debug() -> bool:
    if not _ATTENTION_DEBUG_STATE["enabled"]:
        return False
    if _ATTENTION_DEBUG_STATE["capture_once"] and _ATTENTION_DEBUG_STATE["captured"]:
        return False
    return True


def _record_self_attention_debug(
    *,
    layer_idx: int,
    query_length: int,
    kv_length: int,
    num_heads: int,
    uses_prefix: bool,
) -> None:
    if not _should_capture_attention_debug():
        return
    _ATTENTION_DEBUG_STATE["self_attn_layers"].append(
        {
            "layer_idx": layer_idx,
            "query_length": query_length,
            "kv_length": kv_length,
            "num_heads": num_heads,
            "uses_prefix": uses_prefix,
            "attention_scores": int(query_length * kv_length * num_heads),
        }
    )


def _record_cross_attention_debug(
    *,
    layer_idx: int,
    query_length: int,
    kv_length: int,
    num_heads: int,
) -> None:
    if not _should_capture_attention_debug():
        return
    _ATTENTION_DEBUG_STATE["cross_attn_layers"].append(
        {
            "layer_idx": layer_idx,
            "query_length": query_length,
            "kv_length": kv_length,
            "num_heads": num_heads,
            "attention_scores": int(query_length * kv_length * num_heads),
        }
    )
    _ATTENTION_DEBUG_STATE["captured"] = True


def _repeat_kv_heads(x: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """Expand grouped KV heads to match the number of query heads."""
    if x.shape[1] == num_attention_heads:
        return x
    if num_attention_heads % x.shape[1] != 0:
        raise ValueError(
            f"Cannot expand {x.shape[1]} KV heads to {num_attention_heads} attention heads."
        )
    repeats = num_attention_heads // x.shape[1]
    return x.repeat_interleave(repeats, dim=1)


class ExpertKVCacheCrossAttention(nn.Module):
    """Cross-attention from diffusion hidden states into a read-only VLM KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Any | None,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache is None or layer_idx >= len(kv_cache):
            return torch.zeros_like(hidden_states)

        key_states, value_states = kv_cache[layer_idx]
        if key_states is None or value_states is None or key_states.numel() == 0:
            return torch.zeros_like(hidden_states)

        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, query_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        key_states = _repeat_kv_heads(key_states, self.num_attention_heads)
        value_states = _repeat_kv_heads(value_states, self.num_attention_heads)
        _record_cross_attention_debug(
            layer_idx=layer_idx,
            query_length=query_length,
            kv_length=key_states.shape[-2],
            num_heads=self.num_attention_heads,
        )

        cross_attention_mask = None
        if attention_mask is not None:
            cross_attention_mask = attention_mask[..., : key_states.shape[-2]]
            cross_attention_mask = cross_attention_mask.to(dtype=query_states.dtype)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=cross_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.o_proj(attn_output)


class Qwen3CrossAttentionDecoderLayer(nn.Module):
    """Wrap a Qwen3 decoder layer with an additional KV-cache cross-attention block."""

    def __init__(
        self,
        base_layer: nn.Module,
        replace_prefix_self_attention: bool = True,
        enable_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = base_layer.hidden_size
        self.attention_type = getattr(base_layer, "attention_type", "full_attention")
        self.layer_idx = base_layer.self_attn.layer_idx
        self.replace_prefix_self_attention = replace_prefix_self_attention
        self.enable_cross_attention = enable_cross_attention

        config = base_layer.self_attn.config
        self.cross_attn_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if enable_cross_attention:
            self.cross_attn = ExpertKVCacheCrossAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                head_dim=base_layer.self_attn.head_dim,
                attention_bias=config.attention_bias,
            )
        else:
            self.cross_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.base_layer.input_layernorm(hidden_states)
        self_attention_mask = attention_mask
        self_attention_cache = past_key_values
        if self.replace_prefix_self_attention:
            self_attention_mask = None
            if attention_mask is not None:
                query_length = hidden_states.shape[1]
                self_attention_mask = attention_mask[..., -query_length:]
            self_attention_cache = None
        query_length = hidden_states.shape[1]
        if self_attention_cache is not None:
            kv_length = self_attention_cache[layer_idx][0].shape[-2]
        else:
            kv_length = query_length
        _record_self_attention_debug(
            layer_idx=self.layer_idx,
            query_length=query_length,
            kv_length=kv_length,
            num_heads=self.base_layer.self_attn.config.num_attention_heads,
            uses_prefix=self_attention_cache is not None,
        )
        hidden_states, _ = self.base_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=self_attention_mask,
            position_ids=position_ids,
            past_key_values=self_attention_cache,
            use_cache=use_cache and self_attention_cache is not None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if self.enable_cross_attention:
            residual = hidden_states
            cross_hidden_states = self.cross_attn_layernorm(hidden_states)
            cross_hidden_states = self.cross_attn(
                hidden_states=cross_hidden_states,
                kv_cache=past_key_values,
                layer_idx=self.layer_idx,
                attention_mask=attention_mask,
            )
            hidden_states = residual + cross_hidden_states

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3SelectivePrefixDecoderLayer(nn.Module):
    """Wrap a Qwen3 decoder layer to optionally keep prefix self-attention per layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        keep_prefix_self_attention: bool,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = base_layer.hidden_size
        self.attention_type = getattr(base_layer, "attention_type", "full_attention")
        self.layer_idx = base_layer.self_attn.layer_idx
        self.keep_prefix_self_attention = keep_prefix_self_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.base_layer.input_layernorm(hidden_states)
        self_attention_mask = attention_mask
        self_attention_cache = past_key_values
        if not self.keep_prefix_self_attention:
            self_attention_mask = None
            if attention_mask is not None:
                query_length = hidden_states.shape[1]
                self_attention_mask = attention_mask[..., -query_length:]
            self_attention_cache = None
        query_length = hidden_states.shape[1]
        if self_attention_cache is not None:
            kv_length = self_attention_cache[self.layer_idx][0].shape[-2]
        else:
            kv_length = query_length
        _record_self_attention_debug(
            layer_idx=self.layer_idx,
            query_length=query_length,
            kv_length=kv_length,
            num_heads=self.base_layer.self_attn.config.num_attention_heads,
            uses_prefix=self_attention_cache is not None,
        )
        hidden_states, _ = self.base_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=self_attention_mask,
            position_ids=position_ids,
            past_key_values=self_attention_cache,
            use_cache=use_cache and self_attention_cache is not None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def resolve_cross_attention_layer_indices(
    num_layers: int,
    spec: str | int | Iterable[int] | None,
) -> list[int]:
    if spec is None:
        return []
    if spec == "all":
        return list(range(num_layers))
    if isinstance(spec, int):
        return [spec]
    indices = sorted({int(idx) for idx in spec})
    invalid = [idx for idx in indices if idx < 0 or idx >= num_layers]
    if invalid:
        raise ValueError(f"Invalid cross-attention layer indices: {invalid}")
    return indices


def attach_expert_cross_attention(
    expert: nn.Module,
    layer_spec: str | int | Iterable[int] | None,
    replace_prefix_self_attention: bool = True,
    prefix_self_attention_layers: str | int | Iterable[int] | None = None,
) -> tuple[list[int], list[int]]:
    layers = getattr(expert, "layers", None)
    if layers is None:
        raise TypeError("Expected expert model to expose a top-level `layers` ModuleList.")

    indices = resolve_cross_attention_layer_indices(len(layers), layer_spec)
    prefix_indices = resolve_cross_attention_layer_indices(len(layers), prefix_self_attention_layers)
    if not indices and not prefix_indices:
        return [], []

    wrapped_indices = list(range(len(layers)))
    cross_attention_enabled = bool(indices)
    for idx in wrapped_indices:
        if cross_attention_enabled:
            layers[idx] = Qwen3CrossAttentionDecoderLayer(
                layers[idx],
                replace_prefix_self_attention=(
                    replace_prefix_self_attention and idx not in prefix_indices
                ),
                enable_cross_attention=idx in indices,
            )
        else:
            layers[idx] = Qwen3SelectivePrefixDecoderLayer(
                layers[idx],
                keep_prefix_self_attention=idx in prefix_indices,
            )
    return indices, wrapped_indices


def remap_expert_state_dict_for_cross_attention(
    state_dict: dict[str, torch.Tensor],
    layer_indices: Iterable[int],
) -> dict[str, torch.Tensor]:
    """Map pretrained expert layer weights into wrapped `base_layer.*` keys."""
    remapped_state_dict = dict(state_dict)
    for idx in layer_indices:
        prefix = f"expert.layers.{idx}."
        wrapped_prefix = f"{prefix}base_layer."
        for key in list(remapped_state_dict.keys()):
            if not key.startswith(prefix):
                continue
            if key.startswith(wrapped_prefix):
                continue
            new_key = wrapped_prefix + key[len(prefix) :]
            remapped_state_dict[new_key] = remapped_state_dict.pop(key)
    return remapped_state_dict
