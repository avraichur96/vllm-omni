# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared backbone composition helpers for Pi-family action models."""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

from vllm_omni.diffusion.models.pi.common import attention

PrefixKV = tuple[torch.Tensor, torch.Tensor]
ModuleInputAligner = Callable[[torch.Tensor, nn.Module], torch.Tensor]


def _keep_input_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Leave module inputs unchanged when the variant needs no dtype bridge."""
    del module
    return tensor


def embed_multimodal_prefix(
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    *,
    embed_image: Callable[[torch.Tensor], torch.Tensor],
    embed_language_tokens: Callable[[torch.Tensor], torch.Tensor],
    expected_num_views: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ordered camera and language embeddings into one prefix.

    Cameras are embedded one slot at a time and retain their configured order.
    Each per-batch camera-validity bit is expanded across that camera's image
    tokens, so a missing camera keeps its fixed token positions while masking
    every token in the slot. Language embeddings and their padding mask follow
    the image slots.

    ``embed_language_tokens`` owns any Gemma embedding scaling. Image tokens
    have already passed through the vision projector and are not scaled here.
    All prefix block markers are false, making the prefix bidirectional; each
    variant declares its own causal boundary when it appends the suffix.

    Camera count remains caller-owned: variants with a fixed deployment layout
    pass ``expected_num_views``; variants without one leave it unset.
    """
    num_views = len(images)
    if len(image_masks) != num_views:
        raise ValueError(
            f"images and image_masks must contain the same number of views, got {num_views} and {len(image_masks)}."
        )
    if expected_num_views is not None and num_views != expected_num_views:
        raise ValueError(f"Expected exactly {expected_num_views} image views, got {num_views}.")

    embeddings: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []

    for image, image_mask in zip(images, image_masks):
        image_embedding = embed_image(image)
        batch_size, num_image_tokens = image_embedding.shape[:2]
        embeddings.append(image_embedding)
        padding_masks.append(image_mask[:, None].expand(batch_size, num_image_tokens))

    language_embedding = embed_language_tokens(lang_tokens)
    embeddings.append(language_embedding)
    padding_masks.append(lang_masks)

    prefix_embeddings = torch.cat(embeddings, dim=1)
    prefix_padding_masks = torch.cat(padding_masks, dim=1)
    prefix_attention_markers = torch.zeros(
        (prefix_padding_masks.shape[0], prefix_embeddings.shape[1]),
        dtype=torch.bool,
        device=prefix_embeddings.device,
    )
    return prefix_embeddings, prefix_padding_masks, prefix_attention_markers


def execute_prefix_layer(
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    paligemma: nn.Module,
    *,
    align_module_input: ModuleInputAligner = _keep_input_dtype,
) -> tuple[torch.Tensor, PrefixKV]:
    """Run one ordinary PaliGemma prefix layer and return its post-RoPE K/V.

    The prefix contains ordered image and language tokens and never sees the
    action expert's timestep conditioning. Both Pi variants therefore use the
    stock Gemma residual path here. ``align_module_input`` keeps dtype policy
    explicit: Pi0 currently preserves inputs, while Pi0.5 aligns inputs with
    each projection for mixed-dtype checkpoints.

    K/V are cached after RoPE with shape
    ``(batch, num_kv_heads, prefix_length, head_dim)``. The model-specific
    suffix executor later combines them with freshly projected suffix K/V.
    """
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]

    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    normalized = align_module_input(normalized, layer.self_attn.q_proj)

    hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(value, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)

    attended = attention.eager_attention(
        query,
        key,
        value,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    attended = attended.transpose(1, 2).reshape(
        query.shape[0],
        -1,
        query.shape[1] * layer.self_attn.head_dim,
    )

    attended = align_module_input(attended, layer.self_attn.o_proj)
    hidden_states = layer.self_attn.o_proj(attended) + residual
    residual = hidden_states

    normalized = layer.post_attention_layernorm(hidden_states)
    normalized = align_module_input(normalized, layer.mlp.up_proj)
    hidden_states = layer.mlp(normalized) + residual
    return hidden_states, (key, value)
