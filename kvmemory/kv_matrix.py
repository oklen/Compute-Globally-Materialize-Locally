"""Shared KV-assembly helpers for the materialization experiments.

encode_block: one plain-causal forward over ascending-position ids -> per-layer (K,V)
for a slice, moved to CPU (uses the transformer core, skips lm_head).
assemble: concatenate served KV blocks at their original positions -> (cache, positions).

These are the only names the experiment scripts import from this module.
"""
from __future__ import annotations

import os

import torch
from transformers import DynamicCache

from kvmemory.llm_hf import _iter_cache_kv

@torch.no_grad()
def encode_block(llm, ids, positions, keep_a=None, keep_b=None):
    """One plain-causal forward over ascending-position ids; returns per-layer (K,V) for the
    [keep_a, keep_b) slice, moved to CPU. Uses the transformer CORE (skips lm_head): we only need
    KV, and materializing 152k-vocab logits per encode was the run's dominant waste."""
    dev = llm.device
    idt = torch.tensor([ids], dtype=torch.long, device=dev)
    post = torch.tensor(positions, dtype=torch.long, device=dev)
    cache = DynamicCache()
    core = getattr(llm.model, "model", llm.model)
    # 2026 unified multimodal wrappers: the text decoder lives at .language_model inside
    # the core; identical to `core` for classic causal LMs (attribute absent -> unchanged).
    core = getattr(core, "language_model", None) or core
    # SPRAG_ENCODE_CHUNK: feed the sequence in causal chunks through the same cache --
    # mathematically identical, peak attention memory chunk*L instead of L^2 (needed where
    # fused kernels reject the model's custom mask and fall back to materializing scores,
    # e.g. Gemma4Unified long-context under tf5 SDPA).
    _chunk = int(os.environ.get("SPRAG_ENCODE_CHUNK", "0"))
    if _chunk and len(ids) > _chunk:
        for a0 in range(0, len(ids), _chunk):
            b0 = min(a0 + _chunk, len(ids))
            core(input_ids=idt[:, a0:b0], past_key_values=cache, use_cache=True,
                 position_ids=post[a0:b0].unsqueeze(0),
                 cache_position=torch.arange(a0, b0, device=dev),
                 attention_mask=torch.ones(1, b0, dtype=torch.long, device=dev))
    else:
        core(input_ids=idt, past_key_values=cache, use_cache=True,
             position_ids=post.unsqueeze(0), cache_position=torch.arange(len(ids), device=dev),
             attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev))
    a = 0 if keep_a is None else keep_a
    b = len(ids) if keep_b is None else keep_b
    # slice on GPU FIRST, then move: whole-sequence CPU copies were the other dominant waste
    return [(K[:, :, a:b].contiguous().cpu(), V[:, :, a:b].contiguous().cpu())
            for _, K, V in _iter_cache_kv(cache)]


def slice_kv(kv, a, b):
    return [(K[:, :, a:b], V[:, :, a:b]) for K, V in kv]


@torch.no_grad()
def assemble(llm, blocks):
    """blocks: list of (kv_slices, positions_list) in ascending-position order -> (cache, pos)."""
    dev = llm.device
    new = DynamicCache()
    nl = len(blocks[0][0])
    for li in range(nl):
        Ks = [b[0][li][0].to(dev, non_blocking=True) for b in blocks]
        Vs = [b[0][li][1].to(dev, non_blocking=True) for b in blocks]
        new.update(torch.cat(Ks, dim=2), torch.cat(Vs, dim=2), li)
    pos = []
    for b in blocks:
        pos.extend(b[1])
    return new, torch.tensor(pos, dtype=torch.long, device=llm.device)
