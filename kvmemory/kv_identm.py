"""kv_identm.py -- kv_ident + top-1/top-2 margin, to characterise first-token mismatches.

Same three checks as kv_ident (tokenization / KV-path / header-split), but for the KV-path
check (B) it also reports, per item, the one-shot path's top-1 vs top-2 logit margin. A
first-token argmax mismatch whose margin is <= the path's max|dlogit| is a bf16 near-tie
(the two paths straddle a decision boundary that is itself within numerical noise); a
mismatch whose margin >> deviation would be a real disagreement. Summary prints, for every
mismatched item, (ref_margin, path_dev) so "tie vs real" is decidable from the numbers.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=/home/tiger \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_identm --items 24 --seed 1
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]


def build(rng, n_events=24):
    D, S, T = rng.sample(NAMES, 3)
    others = [n for n in NAMES if n not in (D, S, T)]
    m = rng.randrange(8, 16)
    special = {0: f"register {D} set to state {rng.choice(STATUS2)}",
               m: f"register {S} set to state {rng.choice(STATUS2)}",
               n_events - 1: f"register {T} mirrors register {S}"}
    body = [special.get(i) or
            f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
            for i in range(n_events)]
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n" for i in range(n_events)]
    q = (f"\n\nQuestion: What state is register {T} in? Answer ONLINE or OFFLINE:")
    return evs, q


@torch.no_grad()
def logits_of(llm, ids_list):
    t = torch.tensor([ids_list], dtype=torch.long, device=llm.device)
    out = llm.model(input_ids=t, past_key_values=DynamicCache(), use_cache=True,
                    position_ids=torch.arange(len(ids_list), device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(len(ids_list), device=llm.device),
                    attention_mask=torch.ones_like(t))
    return out.logits[0, -1].float()


@torch.no_grad()
def logits_via_kv(llm, H_ids, flat, qids, split_header=True):
    H = len(H_ids)
    total = H + len(flat)
    if split_header:
        H_kv = encode_block(llm, H_ids, list(range(H)))
        body = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
        c, pos = assemble(llm, [(H_kv, list(range(H))), (body, list(range(H, total)))])
    else:
        allkv = encode_block(llm, H_ids + flat, list(range(total)))
        c, pos = assemble(llm, [(allkv, list(range(total)))])
    L = len(qids)
    nxt = int(pos.max().item()) + 1
    t = torch.tensor([qids], dtype=torch.long, device=llm.device)
    out = llm.model(input_ids=t, past_key_values=c, use_cache=True,
                    position_ids=torch.arange(nxt, nxt + L, device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(total, total + L, device=llm.device),
                    attention_mask=torch.ones(1, total + L, dtype=torch.long, device=llm.device))
    r = out.logits[0, -1].float()
    del c
    return r


def margin(logits):
    top2 = logits.topk(2).values
    return float(top2[0] - top2[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    rng = random.Random(args.seed)

    tok_eq = 0
    devB, argB = [], 0
    margins = []            # one-shot path top1-top2 margin, every item
    mismatches = []         # (it, ref_margin, path_dev) for argmax-mismatched items
    for it in range(args.items):
        evs, q = build(rng)
        H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
        flat = [t for e in evs
                for t in llm.tok(e, add_special_tokens=False).input_ids]
        qids = list(llm.tok(q + tail, add_special_tokens=False).input_ids)
        piecewise = H_ids + flat + qids
        oneshot = list(llm.tok(head_full + "".join(evs) + q + tail,
                               add_special_tokens=False).input_ids)
        tok_eq += int(piecewise == oneshot)

        ref = logits_of(llm, piecewise)
        kv = logits_via_kv(llm, H_ids, flat, qids, split_header=True)
        dev = float((ref - kv).abs().max())
        devB.append(dev)
        rmar = margin(ref)
        margins.append(rmar)
        match = int(int(ref.argmax()) == int(kv.argmax()))
        argB += match
        if not match:
            mismatches.append((it, rmar, dev))
        torch.cuda.empty_cache()
        print(f"[idm] it{it} tok_eq={int(piecewise == oneshot)} argmatch={match} "
              f"ref_margin={rmar:.4f} path_dev={dev:.4f}", flush=True)

    n = args.items
    print("\n===== IDENTITY + MARGIN =====")
    print(f"A. tokenization piecewise == one-shot : {tok_eq}/{n}")
    print(f"B. KV path vs plain prefill (SAME ids): argmax {argB}/{n}, "
          f"max|dlogit| mean={sum(devB)/n:.4f} worst={max(devB):.4f}")
    print(f"   one-shot top1-top2 margin: mean={sum(margins)/n:.4f} "
          f"min={min(margins):.4f} max={max(margins):.4f}")
    if mismatches:
        for it, mar, dev in mismatches:
            verdict = "near-tie (margin<=dev)" if mar <= dev else "REAL disagreement (margin>dev)"
            print(f"   MISMATCH it{it}: ref_margin={mar:.4f} path_dev={dev:.4f} -> {verdict}")
    else:
        print("   no argmax mismatches")
    print("IDENT_DONE", flush=True)


if __name__ == "__main__":
    main()
