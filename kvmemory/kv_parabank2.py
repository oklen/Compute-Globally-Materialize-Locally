"""kv_parabank2.py -- length- and background-controlled rerun of the 16-construction write bank.

kv_parabank draws its filler events INSIDE the per-construction loop, so each construction sees a
different omitted background; and because the carrier is the last retained row, constructions of
different token length also move the appended query. This version removes both:

  * fillers, decoy, mention and names are drawn ONCE per item and shared by all 16 constructions;
  * the carrier event is padded (trailing newlines) to a common token budget = max over the bank,
    so every construction occupies the SAME carrier slot and the query lands at the SAME
    absolute position.

Only the carrier's surface wording varies. Readout is unchanged from kv_parabank (trace prompt +
menu + 160 tokens + pick-last), so the rates are directly comparable to the reported bank.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_parabank2 --items 32 --seed 8000 --out ./out/pb2.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES
from kvmemory.kv_parabank import (STATUS2, N_BODY, DEC_IDX, CAR_IDX, TRACE,
                                  BANK_MIRROR, BANK_FLAG, ev, pickw_last, build)


def run2(llm, H_ids, H, H_kv, tail, evs, q, menu, car_budget, pad_id):
    """As kv_parabank.run2, but the carrier event is padded to `car_budget` tokens so the
    served carrier slot -- and therefore the appended query position -- is identical across
    constructions."""
    eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
    pad = car_budget - len(eids[CAR_IDX])
    if pad < 0:
        raise ValueError(f"carrier longer than budget by {-pad}")
    eids[CAR_IDX] = eids[CAR_IDX] + [pad_id] * pad
    spans, cur = [], H
    for e in eids:
        spans.append((cur, cur + len(e)))
        cur += len(e)
    flat = [t for e in eids for t in e]
    full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
    outs = []
    for served in ([DEC_IDX, CAR_IDX], list(range(len(evs)))):
        crows = [pp for i in served for pp in range(*spans[i])]
        rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
        sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
        c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
        o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 160)
        outs.append(pickw_last(o, menu))
        del c, sub
    del full_kv
    torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--out", default="./out/pb2.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head + HDR, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    pad_id = llm.tok("\n", add_special_tokens=False).input_ids[-1]
    tlen = lambda t: len(llm.tok(t, add_special_tokens=False).input_ids)
    rng = random.Random(args.seed)
    rows = []

    for it in range(args.items):
        # ================= mirror family =================
        D, S, T = rng.sample(NAMES, 3)
        gold = rng.choice(STATUS2)
        m = rng.randrange(8, 16)
        qm = (f"\n\nQuestion: What state is register {T} in? "
              "Answer ONLINE or OFFLINE." + TRACE)
        # ONE background draw, shared by all eight constructions
        base = build(rng, {DEC_IDX: f"register {D} completed a routine self-test",
                           m: f"register {S} set to state {gold}",
                           CAR_IDX: "", "_names": [D, S, T]})
        car = {k: ev(CAR_IDX, t.format(a=T, b=S)) for k, t in BANK_MIRROR}
        budget = max(tlen(t) for t in car.values())
        for key, _ in BANK_MIRROR:
            evs = list(base)
            evs[CAR_IDX] = car[key]
            kv_a, txt_a = run2(llm, H_ids, H, H_kv, tail, evs, qm, STATUS2, budget, pad_id)
            rows.append({"it": it, "fam": "mirror", "key": key, "gold": gold,
                         "kv_a": kv_a, "txt_a": txt_a, "budget": budget,
                         "kv": int(kv_a == gold), "txt": int(txt_a == gold)})

        # ================= flag family =================
        D2, S2 = rng.sample(NAMES, 2)
        gv = rng.randrange(100, 999)
        thr = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            thr = rng.randrange(100, 999)
        vgold = "EXCEEDED" if gv > thr else "NORMAL"
        m2 = rng.randrange(8, 14)
        qf = (f"\n\nQuestion: Was sensor {S2} above the alert threshold? "
              "Answer EXCEEDED or NORMAL." + TRACE)
        base2 = build(rng, {DEC_IDX: f"register {D2} completed a routine self-test",
                            m2: f"sensor {S2} measured {gv}; the alert threshold is {thr}",
                            m2 + 1: f"alert check result for sensor {S2}: threshold {vgold}",
                            CAR_IDX: "", "_names": [D2, S2]})
        car2 = {k: ev(CAR_IDX, t.format(r=S2)) for k, t in BANK_FLAG}
        budget2 = max(tlen(t) for t in car2.values())
        for key, _ in BANK_FLAG:
            evs = list(base2)
            evs[CAR_IDX] = car2[key]
            kv_a, txt_a = run2(llm, H_ids, H, H_kv, tail, evs, qf,
                               ["EXCEEDED", "NORMAL"], budget2, pad_id)
            rows.append({"it": it, "fam": "flag", "key": key, "gold": vgold,
                         "kv_a": kv_a, "txt_a": txt_a, "budget": budget2,
                         "kv": int(kv_a == vgold), "txt": int(txt_a == vgold)})

        dm = [r for r in rows if r["it"] == it and r["fam"] == "mirror"]
        df = [r for r in rows if r["it"] == it and r["fam"] == "flag"]
        print(f"[pb2] it{it} mirror kv={''.join(str(r['kv']) for r in dm)} "
              f"flag kv={''.join(str(r['kv']) for r in df)} "
              f"budgets={dm[0]['budget']}/{df[0]['budget']}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)

    print("\n===== LENGTH-CONTROLLED BANK (kv / txt per construction) =====")
    for fam, bank in (("mirror", BANK_MIRROR), ("flag", BANK_FLAG)):
        for key, _ in bank:
            rr = [r for r in rows if r["key"] == key]
            if not rr:
                continue
            kv = sum(r["kv"] for r in rr) / len(rr)
            tx = sum(r["txt"] for r in rr) / len(rr)
            print(f"  {fam:6s} {key:9s} n={len(rr):3d} kv={kv:.3f} txt={tx:.3f}")
    print("PB2_DONE")


if __name__ == "__main__":
    main()
