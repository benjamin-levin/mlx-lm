# Copyright (c) 2026 Apple Inc.
"""SnapKV: content-aware KV-cache compression for long-context inference.

SnapKV runs the prompt through the model once, scores each cached K position
using the attention scores from the last ``obs_window`` queries, then
physically drops the un-selected K/V entries before the decode loop starts.
The result is a small, fixed-shape cache that streams at "StreamingLLM speed"
while preserving the high-attention content the model would actually need
to attend to during decode.

This is opt-in and gated on prompt length: SnapKV adds prefill overhead and
only pays off for prompts long enough that decode dominates total latency.
On Qwen3-Next at 95k context (M4 Max 36GB), SnapKV gave 1.31x end-to-end
speedup with 3/3 retrieval pass at top_k=4096.

Reference paper: "SnapKV: LLM Knows What You are Looking for Before
Generation", Li et al., 2024 (https://arxiv.org/abs/2404.14469).

Usage::

    from mlx_lm.snapkv import patch_for_snapkv, snapkv_prefill_and_trim

    patch_for_snapkv(model)   # class-level monkey-patch on Qwen3NextAttention
    prompt_cache, last_logits = snapkv_prefill_and_trim(
        model, prompt_ids,
        top_k=4096, n_sink=128, n_window=512,
        obs_window=32, pool_kernel=1,
    )
    # decode normally with prompt_cache; SnapKVCache instances have replaced
    # the full-attention layers' KVCache entries.

Or via ``generate_step`` opt-in (see ``mlx_lm.generate.generate_step``).

Currently supports the Qwen3-Next family (the only model where this is
bench-validated upstream). Adding another model only requires that its
attention class share the Qwen3Next attention signature: ``__call__(self,
x, mask=None, cache=None)`` with a post-RoPE query path.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .models.base import scaled_dot_product_attention
from .models.cache import KVCache, SnapKVCache, make_prompt_cache

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patched attention call: capture last ``obs_window`` queries post-RoPE.
# ---------------------------------------------------------------------------


def _snapkv_attention_call(self, x, mask=None, cache=None):
    """Drop-in for ``Qwen3NextAttention.__call__`` that captures the last
    ``self._snapkv_obs_window`` post-RoPE queries when capture is enabled.

    Behavior is bit-identical to the original when ``_snapkv_capture`` is
    False (the default), so it is safe to install on the class once and
    selectively activate per-instance.
    """
    B, L, _ = x.shape

    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)

    keys, values = self.k_proj(x), self.v_proj(x)

    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    # SnapKV capture: only during prefill (L > 1) and only when explicitly on.
    if getattr(self, "_snapkv_capture", False) and L > 1:
        obs = int(getattr(self, "_snapkv_obs_window", 32))
        self._snapkv_last_queries = queries[..., -obs:, :] if L >= obs else queries

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


def _full_attn_layers(model):
    """Return the list of full-attention DecoderLayer objects."""
    tm = (
        getattr(model, "language_model", None) or getattr(model, "model", None) or model
    )
    out = []
    for lyr in tm.layers:
        if getattr(lyr, "is_linear", False):
            continue
        if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "q_proj"):
            out.append(lyr)
    return out


def _attn_cache_indices(model, prompt_cache):
    tm = (
        getattr(model, "language_model", None) or getattr(model, "model", None) or model
    )
    out = []
    for i, lyr in enumerate(tm.layers):
        if getattr(lyr, "is_linear", False):
            continue
        if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "q_proj"):
            out.append(i)
    return out


def patch_for_snapkv(model, *, obs_window: int = 32) -> int:
    """Install the SnapKV query-capture hook on the model's attention class.

    Patches ``Qwen3NextAttention.__call__`` at the **class** level, not the
    instance level: Python dispatches ``obj(x)`` via ``type(obj).__call__``,
    so an instance-level assignment would silently never fire. The patched
    callable is a transparent no-op when ``_snapkv_capture`` is False.

    Args:
        model: A loaded MLX model. Must expose a Qwen3-Next attention
            layer (``hasattr(lyr.self_attn, "q_proj")`` on every non-linear
            decoder layer).
        obs_window: Number of trailing prompt queries used to score
            positions during the trim step. SnapKV paper recommends 32.

    Returns:
        The number of full-attention layers initialized.

    Raises:
        ImportError: if ``mlx_lm.models.qwen3_next`` is not importable
            (e.g. the model is not Qwen3-Next).
    """
    from .models.qwen3_next import Qwen3NextAttention

    if not hasattr(Qwen3NextAttention, "_snapkv_orig_call"):
        Qwen3NextAttention._snapkv_orig_call = Qwen3NextAttention.__call__
        Qwen3NextAttention.__call__ = _snapkv_attention_call

    n = 0
    for lyr in _full_attn_layers(model):
        attn = lyr.self_attn
        attn._snapkv_obs_window = int(obs_window)
        attn._snapkv_capture = False
        attn._snapkv_last_queries = None
        n += 1
    _LOG.info(
        "SnapKV: patched Qwen3NextAttention class, initialized %d full-attn layers "
        "(obs_window=%d)",
        n,
        obs_window,
    )
    return n


def unpatch_snapkv(model) -> int:
    """Restore the original class-level ``__call__`` on Qwen3NextAttention."""
    from .models.qwen3_next import Qwen3NextAttention

    if hasattr(Qwen3NextAttention, "_snapkv_orig_call"):
        Qwen3NextAttention.__call__ = Qwen3NextAttention._snapkv_orig_call
        del Qwen3NextAttention._snapkv_orig_call
        return 1
    return 0


def _set_capture(model, on: bool) -> None:
    for lyr in _full_attn_layers(model):
        lyr.self_attn._snapkv_capture = bool(on)


# ---------------------------------------------------------------------------
# Position selection: score, pool, top-K.
# ---------------------------------------------------------------------------


def _snapkv_select_indices(
    queries: mx.array,  # (B, H, obs_window, D), post-RoPE
    keys: mx.array,  # (B, KV_H, T, D), post-RoPE
    *,
    top_k: int,
    n_sink: int,
    n_window: int,
    pool_kernel: int,
    scale: float,
) -> mx.array:
    """Return int32 indices (sorted ascending) of positions to KEEP.

    Heuristic from the SnapKV paper:
      - the first ``n_sink`` positions are always kept (attention sink),
      - the last ``n_window`` positions are always kept (recent tokens),
      - in the middle, keep the ``top_k`` positions with the highest
        max attention score across the observation-window queries and
        across heads.
    """
    B, H, OW, D = queries.shape
    _, KV_H, T, _ = keys.shape
    n_repeats = H // KV_H

    q_g = queries.reshape(B, KV_H, n_repeats, OW, D)
    scores = mx.einsum("bhrqd,bhtd->bhrqt", q_g, keys) * scale
    scores = scores.max(axis=(2, 3))  # (B, KV_H, T)
    scores = scores.max(axis=1)  # (B, T)
    scores = scores[0]  # (T,)

    if pool_kernel > 1:
        # Box-filter pooling via cumsum, valid-padded.
        c = mx.cumsum(scores, axis=0)
        pad = pool_kernel // 2
        c = mx.concatenate([mx.zeros((1,), dtype=c.dtype), c])
        lo = mx.maximum(mx.arange(T) - pad, mx.array(0))
        hi = mx.minimum(mx.arange(T) + pad + 1, mx.array(T))
        scores = (c[hi] - c[lo]) / (hi - lo).astype(c.dtype)

    mid_lo, mid_hi = n_sink, T - n_window
    if mid_hi <= mid_lo:
        return mx.arange(T, dtype=mx.int32)

    mid_scores = scores[mid_lo:mid_hi]
    k_to_pick = int(min(top_k, mid_scores.shape[0]))
    if k_to_pick <= 0:
        keep = mx.concatenate(
            [
                mx.arange(n_sink, dtype=mx.int32),
                mx.arange(T - n_window, T, dtype=mx.int32),
            ]
        )
        return mx.sort(keep)

    neg = -mid_scores
    idx = mx.argpartition(neg, kth=k_to_pick - 1, axis=-1)[..., :k_to_pick]
    idx = idx.astype(mx.int32) + mx.array(mid_lo, dtype=mx.int32)
    sink = mx.arange(n_sink, dtype=mx.int32)
    win = mx.arange(T - n_window, T, dtype=mx.int32)
    return mx.sort(mx.concatenate([sink, idx, win]))


# ---------------------------------------------------------------------------
# Driver: prefill + trim. Decode is then ordinary mlx_lm generate.
# ---------------------------------------------------------------------------


def snapkv_prefill_and_trim(
    model,
    prompt_ids,
    *,
    top_k: int = 4096,
    n_sink: int = 128,
    n_window: int = 512,
    obs_window: int = 32,
    pool_kernel: int = 1,
    prefill_chunk: int = 2048,
):
    """Prefill ``prompt_ids`` through ``model``, then trim each
    full-attention layer's cache to (``n_sink + top_k + n_window``)
    positions via SnapKV scoring.

    The caller is responsible for invoking :func:`patch_for_snapkv`
    beforehand. This function manages the capture flag.

    Args:
        model: Loaded MLX model.
        prompt_ids: A 1-D array-like of token ids (list or ``mx.array``).
        top_k: Number of mid-prompt positions to keep per layer.
        n_sink: Number of initial positions to always keep.
        n_window: Number of trailing positions to always keep (also
            forms the sliding-decode region of the resulting SnapKVCache).
        obs_window: Number of trailing prompt queries used for scoring.
            Must match what was passed to :func:`patch_for_snapkv`.
        pool_kernel: Box-filter width applied to per-position scores
            (1 disables pooling; the paper uses 5-7).
        prefill_chunk: Token-chunk size to bound prefill activation memory.

    Returns:
        ``(prompt_cache, last_logits)`` where ``prompt_cache`` is a list
        of cache objects with full-attention entries replaced by
        :class:`mlx_lm.models.cache.SnapKVCache` instances, and
        ``last_logits`` has shape ``(1, V)`` — the logits at the position
        immediately after the prompt.
    """
    if not isinstance(prompt_ids, mx.array):
        prompt_ids = list(prompt_ids)

    pc = make_prompt_cache(model)

    n = (
        len(prompt_ids)
        if not isinstance(prompt_ids, mx.array)
        else int(prompt_ids.shape[0])
    )
    if n == 0:
        raise ValueError("snapkv_prefill_and_trim: empty prompt")

    _set_capture(model, True)
    logits = None
    i = 0
    while i < n:
        j = min(n, i + prefill_chunk)
        if isinstance(prompt_ids, mx.array):
            chunk = prompt_ids[i:j][None]
        else:
            chunk = mx.array(prompt_ids[i:j])[None]
        logits = model(chunk, cache=pc)
        mx.eval(logits)
        i = j
    _set_capture(model, False)

    last_logits = logits[:, -1, :]

    full_layers = _full_attn_layers(model)
    cache_indices = _attn_cache_indices(model, pc)
    n_trimmed = 0
    n_orig_total = 0
    n_kept_total = 0
    for lyr, cache_idx in zip(full_layers, cache_indices):
        attn = lyr.self_attn
        captured_q = getattr(attn, "_snapkv_last_queries", None)
        if captured_q is None:
            continue
        cache_obj = pc[cache_idx]
        if not isinstance(cache_obj, KVCache):
            continue
        k_full = cache_obj.keys[..., : cache_obj.offset, :]
        v_full = cache_obj.values[..., : cache_obj.offset, :]
        n_orig_total += k_full.shape[-2]
        keep_idx = _snapkv_select_indices(
            captured_q,
            k_full,
            top_k=top_k,
            n_sink=n_sink,
            n_window=n_window,
            pool_kernel=pool_kernel,
            scale=attn.scale,
        )
        keep_idx_b = mx.broadcast_to(
            keep_idx[None, None, :, None],
            (k_full.shape[0], k_full.shape[1], keep_idx.shape[0], k_full.shape[3]),
        )
        new_k = mx.take_along_axis(k_full, keep_idx_b, axis=-2)
        new_v = mx.take_along_axis(v_full, keep_idx_b, axis=-2)
        mx.eval(new_k, new_v)
        n_kept_total += new_k.shape[-2]
        n_pin_layer = max(0, new_k.shape[-2] - n_window)
        pc[cache_idx] = SnapKVCache(
            new_k,
            new_v,
            logical_offset=cache_obj.offset,
            n_pin=n_pin_layer,
        )
        n_trimmed += 1
        attn._snapkv_last_queries = None

    _LOG.info(
        "SnapKV trimmed %d full-attn caches: avg %d -> %d positions",
        n_trimmed,
        n_orig_total // max(1, n_trimmed),
        n_kept_total // max(1, n_trimmed),
    )
    return pc, last_logits
