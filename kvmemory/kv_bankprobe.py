"""kv_bankprobe.py -- self-audit G1: is "understands-but-no-write-through" a generation-
interface artifact, or is the information absent from the readout distribution itself?

kv_parabank's headline (Llama: txt 1.00, KV arm silent in 46-63% of items) rests on ONE
generative readout. Before the paper claims native inaccessibility, probe the same cells at
the LOGIT level: force the "Answer:" position and read the candidate-word probability mass
directly (machinery = kv_fail2.answer_with_probs). If the candidate-restricted argmax is also
at chance, inaccessibility holds at the distribution level; if it recovers well above chance,
the claim must be demoted to "not accessible via free generation" (storage-vs-interface
discipline applied to our own newest result).

Cells: 4 anchor constructions (mirror: mirrors/tracks; flag: accord/consist), KV arm
(serve {DEC, CAR}) with three readouts (trace-160/pick-last, direct-10/pick-first,
forced-"Answer:" candidate-logit probe) + txt arm (serve all) with the probe readout as the
"model knows it here" margin calibration.

    SPRAG_MODEL_PATH=/path/to/Llama31-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_bankprobe --items 1 --seed 999 --out ./out/bp.json
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
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_parabank import BANK_MIRROR, BANK_FLAG

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
TRACE = (" Locate the relevant statement, then end your answer with the final state word.")
BANK4 = [
    ("mirror", "mirrors", "register {a} mirrors register {b}"),
    ("mirror", "tracks", "register {a} tracks register {b}"),
    ("flag", "accord", "the alert status flag for sensor {r} was set according to the check result"),
    ("flag", "consist", "the alert flag for sensor {r} was made consistent with the check result"),
]


def pickw(o, words, last=False):
    u = o.upper()
    f = (lambda w: u.rfind(w)) if last else (lambda w: u.find(w))
    hits = [(f(w), w) for w in words if f(w) >= 0]
    if not hits:
        return None
    return max(hits)[1] if last else min(hits)[1]


def build(rng, specials, n=N_BODY):
    others = [x for x in NAMES if x not in specials.get("_names", [])]
    body = {i: t for i, t in specials.items() if i != "_names"}
    for i in range(n):
        if i not in body:
            body[i] = f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
    return [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--bank", default="anchor4", choices=["anchor4", "full16"])
    ap.add_argument("--seed", type=int, default=8800)
    ap.add_argument("--out", default="./out/bp.json")
    args = ap.parse_args()
    global BANK4
    if args.bank == "full16":
        BANK4 = ([("mirror", k, t) for k, t in BANK_MIRROR]
                 + [("flag", k, t) for k, t in BANK_FLAG])
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    rows = []

    for it in range(args.items):
        D, S, T = rng.sample(NAMES, 3)
        gold_m = rng.choice(STATUS2)
        m = rng.randrange(8, 16)
        D2, S2 = rng.sample(NAMES, 2)
        gv = rng.randrange(100, 999)
        thr = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            thr = rng.randrange(100, 999)
        gold_f = "EXCEEDED" if gv > thr else "NORMAL"
        m2 = rng.randrange(8, 14)

        for fam, key, ctpl in BANK4:
            if fam == "mirror":
                sp = {DEC_IDX: f"register {D} completed a routine self-test",
                      m: f"register {S} set to state {gold_m}",
                      CAR_IDX: ctpl.format(a=T, b=S), "_names": [D, S, T]}
                menu, gold = STATUS2, gold_m
                qbase = (f"\n\nQuestion: What state is register {T} in? "
                         f"Answer {menu[0]} or {menu[1]}.")
            else:
                sp = {DEC_IDX: f"register {D2} completed a routine self-test",
                      m2: f"sensor {S2} measured {gv}; the alert threshold is {thr}",
                      m2 + 1: f"alert check result for sensor {S2}: threshold {gold_f}",
                      CAR_IDX: ctpl.format(r=S2), "_names": [D2, S2]}
                menu, gold = ["EXCEEDED", "NORMAL"], gold_f
                qbase = (f"\n\nQuestion: Was sensor {S2} above the alert threshold? "
                         f"Answer {menu[0]} or {menu[1]}.")
            evs = build(rng, sp)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            flat = [t for e in eids for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
            rec = {"it": it, "fam": fam, "key": key, "gold": gold}
            for arm, served in (("kv", [DEC_IDX, CAR_IDX]), ("txt", list(range(len(evs))))):
                crows = [pp for i in served for pp in range(*spans[i])]
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]

                def fresh():
                    return assemble(llm, [(H_kv, list(range(H))), (sub, crows)])

                # one assemble per readout: generation extends the cache in place
                if arm == "kv":
                    c, p = fresh()
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(qbase + TRACE + tail), 160)
                    rec["kv_trace"] = int(pickw(o, menu, last=True) == gold)
                    del c
                    c, p = fresh()
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(qbase + tail), 10)
                    rec["kv_direct"] = int(pickw(o, menu) == gold)
                    del c
                c, p = fresh()
                raw, cp = answer_with_probs(llm, c, p,
                                            llm._ids(qbase + tail + "Answer:"), menu)
                alt = [w for w in menu if w != gold][0]
                pg, pa = cp.get(gold, 0.0), cp.get(alt, 0.0)
                rec[f"{arm}_p_gold"] = round(pg, 5)
                rec[f"{arm}_p_alt"] = round(pa, 5)
                rec[f"{arm}_probe_ok"] = int(pg > pa)
                del c, sub
            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
        r4 = rows[-4:]
        print(f"[bp] it{it} " + " ".join(
            f"{r['key']}:t{r['kv_trace']}d{r['kv_direct']}p{r['kv_probe_ok']}"
            f"({r['kv_p_gold']:.2f}/{r['kv_p_alt']:.2f})T{r['txt_probe_ok']}" for r in r4),
            flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("\n===== BANK PROBE (per construction) =====")
    for fam, key, _ in BANK4:
        g = [r for r in rows if r["key"] == key]
        n = len(g)
        print(f"{fam:6s} {key:8s} trace={sum(r['kv_trace'] for r in g)/n:.2f} "
              f"direct={sum(r['kv_direct'] for r in g)/n:.2f} "
              f"probe={sum(r['kv_probe_ok'] for r in g)/n:.2f} "
              f"kv_mass={sum(r['kv_p_gold']+r['kv_p_alt'] for r in g)/n:.2f} "
              f"txt_probe={sum(r['txt_probe_ok'] for r in g)/n:.2f}")
    print("BP_DONE", flush=True)


if __name__ == "__main__":
    main()
