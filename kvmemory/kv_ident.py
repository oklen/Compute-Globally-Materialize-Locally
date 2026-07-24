"""kv_ident.py -- is the encode_block/assemble KV path numerically faithful to a plain prefill?

kv_cogate's implementation check reported harvest-all == full prefill on only 189/256 items. If
that is a real defect in the KV path it contaminates EVERY result on this line (kv_decoyctl's
+0.379, E4's boundary map, the coherence pilot). Before anything else, find out.

The prime suspect is the CHECK, not the path: cogate compared

    o1 = assemble(H_kv, full_kv) -> _greedy_pos(c, pos, qids)          # ids tokenized PIECEWISE
    o2 = _greedy(fresh, 0, _ids(head + "".join(evs) + q + tail))       # ids tokenized ONE-SHOT

and BPE merges across string boundaries, so the two arms may not even be running the same token
sequence. This isolates the three candidate causes by holding everything else fixed:

  A. TOKENIZE   piecewise (H_ids + flat + qids) vs one-shot _ids(whole string) -- are the id
                sequences literally equal? If not, cogate's 189/256 is explained and the KV path
                is exonerated by (B).
  B. KV PATH    same ids both sides: fresh full prefill vs encode_block -> assemble ->
                _greedy_pos. Any mismatch here is a REAL bug and everything upstream is void.
  C. SPLIT      header encoded alone (as every experiment does) vs header encoded as part of the
                body forward -- confirms the header rows are prefix-independent, as causality says.

Reports greedy-output agreement AND max |logit| deviation, since a greedy match can hide drift.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=/home/tiger \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_ident --items 24 --seed 1
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
    """Plain full prefill over an explicit id list -> last-position logits."""
    t = torch.tensor([ids_list], dtype=torch.long, device=llm.device)
    out = llm.model(input_ids=t, past_key_values=DynamicCache(), use_cache=True,
                    position_ids=torch.arange(len(ids_list), device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(len(ids_list), device=llm.device),
                    attention_mask=torch.ones_like(t))
    return out.logits[0, -1].float()


@torch.no_grad()
def logits_via_kv(llm, H_ids, flat, qids, split_header=True):
    """encode_block -> assemble -> one forward of qids at explicit positions."""
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
    devB, devC, argB, argC = [], [], 0, 0
    for it in range(args.items):
        evs, q = build(rng)
        H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
        flat = [t for e in evs
                for t in llm.tok(e, add_special_tokens=False).input_ids]
        qids = list(llm.tok(q + tail, add_special_tokens=False).input_ids)
        piecewise = H_ids + flat + qids
        oneshot = list(llm.tok(head_full + "".join(evs) + q + tail,
                               add_special_tokens=False).input_ids)

        # A. tokenization
        tok_eq += int(piecewise == oneshot)

        # B. KV path, SAME ids on both sides
        ref = logits_of(llm, piecewise)
        kv = logits_via_kv(llm, H_ids, flat, qids, split_header=True)
        devB.append(float((ref - kv).abs().max()))
        argB += int(int(ref.argmax()) == int(kv.argmax()))

        # C. header encoded alone vs encoded inside the body forward
        kv2 = logits_via_kv(llm, H_ids, flat, qids, split_header=False)
        devC.append(float((kv - kv2).abs().max()))
        argC += int(int(kv.argmax()) == int(kv2.argmax()))
        torch.cuda.empty_cache()
        if it == 0:
            print(f"  len piecewise={len(piecewise)} oneshot={len(oneshot)}")
            if piecewise != oneshot:
                d = next((i for i in range(min(len(piecewise), len(oneshot)))
                          if piecewise[i] != oneshot[i]), None)
                if d is not None:
                    print(f"  first divergence at token {d}: "
                          f"piecewise={piecewise[d-2:d+3]} oneshot={oneshot[d-2:d+3]}")
                    print(f"    piecewise ctx: {llm.tok.decode(piecewise[max(0,d-6):d+6])!r}")
                    print(f"    oneshot   ctx: {llm.tok.decode(oneshot[max(0,d-6):d+6])!r}")
        print(f"[id] it{it} tok_eq={int(piecewise == oneshot)} "
              f"B_maxdev={devB[-1]:.4f} B_argmax={argB} C_maxdev={devC[-1]:.4f}", flush=True)

    n = args.items
    print("\n===== IDENTITY =====")
    print(f"A. tokenization piecewise == one-shot : {tok_eq}/{n}"
          + ("   <-- explains cogate's 189/256" if tok_eq < n else ""))
    print(f"B. KV path vs plain prefill (SAME ids): argmax {argB}/{n}, "
          f"max|dlogit| mean={sum(devB)/n:.4f} worst={max(devB):.4f}"
          + ("   <-- REAL BUG, everything upstream is void" if argB < n else "   <-- clean"))
    print(f"C. header-alone vs header-in-body      : argmax {argC}/{n}, "
          f"max|dlogit| mean={sum(devC)/n:.4f} worst={max(devC):.4f}")
    print("IDENT_DONE", flush=True)


if __name__ == "__main__":
    main()
