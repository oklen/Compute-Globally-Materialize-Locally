"""kv_fail2.py -- readout-failure case study, FIXED confidence probe: when the row can't be read out, what does the
model actually DO -- guess flat (乱蒙), echo the served label, anti-echo it, or refuse?

Aggregate scores so far only hint at this, and the hints conflict: at K=4 (menu) failures echo
the served decoy 92% of the time; at K=2 (open-ended) the two-parameter fit said failures
answer the decoy's OPPOSITE. This run saves, per item: the raw generation AND the probability
of each candidate answer's first token at the first decode position -- so failures can be
classified (echo / anti-echo / other / refuse) and their CONFIDENCE measured (near-tie margin
= guessing; large margin on the wrong word = systematic bias).

Conditions (each the standard 24-event log, source dropped, {decoy-event, carrier} served):
  bin2_menu   binary state, menu in question       -- mostly succeeds (~.93): success contrast
  bin2_open   binary state, open-ended question    -- the r~0.80 regime: ~20% failures
  dir4_menu   4-way state, menu                    -- the ~78%-failure echo regime
  verdict     operands dropped, carrier reviews S; menu EXCEEDED/NORMAL -- computed-not-stored
              (chance): here the decoy event has NO answer-class word, so echo is impossible --
              what fills the vacuum?
  neutral     upstream verdict explicitly stated then dropped; carrier neutral ("checkpoint
              saved") -- the nothing-bound regime (chance): same question, different vacuum

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_fail2 --items 32 --seed 2700 --out ./out/k2.json
"""
from __future__ import annotations

import argparse
import json
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
DIR4 = ["NORTH", "SOUTH", "EAST", "WEST"]
VERD = ["EXCEEDED", "NORMAL"]
CONDS = ["bin2_menu", "bin2_open", "dir4_menu", "verdict", "neutral"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23


@torch.no_grad()
def answer_with_probs(llm, cache, pos, qids, cand_words, ntok=16):
    """kv_fail's probe read the FIRST decode position, where the model emits "Answer" -- the
    candidate mass there is ~0 and the readings were garbage. Fix: append "Answer:" to the
    prompt itself, so the next-token distribution IS the answer-word distribution; sum the
    with-space and without-space first-token variants of each candidate. Greedy continuation
    from there gives the raw text for the behavioral cross-check."""
    dev = llm.device
    cache_len = pos.shape[0]
    L = qids.shape[1]
    nxt_pos = int(pos.max().item()) + 1
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(nxt_pos, nxt_pos + L, device=dev).unsqueeze(0),
                    cache_position=torch.arange(cache_len, cache_len + L, device=dev),
                    attention_mask=torch.ones(1, cache_len + L, dtype=torch.long, device=dev))
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    cand = {}
    for w in cand_words:
        tot = 0.0
        seen = set()
        for v in (" " + w, w):
            t0 = llm.tok(v, add_special_tokens=False).input_ids[0]
            if t0 not in seen:
                seen.add(t0)
                tot += float(probs[t0])
        cand[w] = tot
    # greedy continuation for the raw text
    nxt = int(logits.argmax())
    cur, cur_pos, toks = cache_len + L, nxt_pos + L, []
    for _ in range(ntok):
        if nxt in llm.eos_ids:
            break
        toks.append(nxt)
        o = llm.model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache,
                      use_cache=True,
                      position_ids=torch.tensor([[cur_pos]], device=dev),
                      cache_position=torch.tensor([cur], device=dev),
                      attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev))
        nxt = int(o.logits[0, -1].argmax())
        cur += 1
        cur_pos += 1
    return llm.tok.decode(toks, skip_special_tokens=True).strip(), cand


def gen_item(rng, cond):
    D, S, T = rng.sample(NAMES, 3)
    m = rng.randrange(8, 16)
    others = [n for n in NAMES if n not in (D, S, T)]
    body = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
            for i in range(N_BODY)}
    if cond in ("bin2_menu", "bin2_open", "dir4_menu"):
        pool = DIR4 if cond == "dir4_menu" else STATUS2
        gold = rng.choice(pool)
        decoy = rng.choice(pool)                    # independent
        body[DEC_IDX] = f"register {D} set to state {decoy}"
        body[m] = f"register {S} set to state {gold}"
        body[CAR_IDX] = f"register {T} mirrors register {S}"
        if cond == "bin2_open":
            q = (f"\n\nQuestion: What state is register {T} in? "
                 "Answer with the single state word only:")
        else:
            menu = " Answer " + ", ".join(pool[:-1]) + f", or {pool[-1]}:" \
                if len(pool) > 2 else f" Answer {pool[0]} or {pool[1]}:"
            q = f"\n\nQuestion: What state is register {T} in?{menu}"
        cands = pool
    else:
        gv = rng.randrange(100, 999)
        thr = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            thr = rng.randrange(100, 999)
        gold = "EXCEEDED" if gv > thr else "NORMAL"
        decoy = None
        body[DEC_IDX] = f"sensor {D} logged a routine idle reading"
        if cond == "verdict":
            body[m] = f"sensor {S} measured {gv}; the alert threshold is {thr}"
            body[CAR_IDX] = f"the operator reviewed sensor {S} against the alert threshold"
        else:  # neutral: verdict explicitly stated upstream, then dropped; carrier neutral
            body[m] = (f"alert check result for sensor {S}: threshold {gold}")
            body[CAR_IDX] = f"checkpoint saved for the sensor {S} review"
        q = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
             "Answer EXCEEDED or NORMAL:")
        cands = VERD
    evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
           for i in range(N_BODY)]
    return evs, q, gold, decoy, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2500)
    ap.add_argument("--out", default="./out/k2.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for cond in CONDS:
        for it in range(args.items):
            evs, q, gold, decoy, cands = gen_item(rng, cond)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            crows = [pp for i in (DEC_IDX, CAR_IDX) for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            raw, cp = answer_with_probs(llm, c, p, llm._ids(q + tail + "Answer:"), cands)
            del c, full_kv, sub
            torch.cuda.empty_cache()
            rows.append({"cond": cond, "it": it, "gold": gold, "decoy": decoy,
                         "raw": raw[:160], "probs": cp})
            print(f"[k2] {cond} it{it} gold={gold[:4]} dec={str(decoy)[:4]} "
                  f"p={ {k: round(v, 2) for k, v in cp.items()} } raw={raw[:28]!r}",
                  flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nn={len(rows)}")
    print("K2_DONE", flush=True)


if __name__ == "__main__":
    main()
