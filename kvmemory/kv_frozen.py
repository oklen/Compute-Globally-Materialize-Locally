"""kv_frozen.py -- frozen-design replication: fresh seeds + an untuned paraphrase template.

Reviewer's hard item #1: core results must survive (a) brand-new seeds and (b) a surface
template that never participated in any tuning. Four core cells, each run under BOTH
templates:

  cell1  binding recall, none-cell     bin2, no state word served, menu    expect ~.95 (8B)
  cell2  binding gradient              explicit verdict upstream dropped;
         carrier neutral/outcome/state; menu EXCEEDED-NORMAL               expect .5/.5/.9
  cell3  accessibility cliff           dir4 native menu                    expect ~chance
  cell4  chain: root/leaf/full         {C1}askM1 / {C3}askT / {all}askT,
         trace readout                                                     expect .7/.5/.7

Template ORIG is the tuned one ("<step i>/action: reg_op()/observation:", "set to state",
"mirrors"/"copied from"). Template PARA is a fresh surface: "<entry i>/event: system_poll()/
result:", "the status of register X was logged as ONLINE", "register T tracks register S",
carriers reworded. Semantics identical; wording untuned.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_frozen --items 16 --seed 5000 --out ./out/fz.json
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
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23
TPLS = ["orig", "para"]
TRACE = (" Locate the relevant statement, then end your answer with the final state word.")


def wrap(tpl, i, body):
    if tpl == "orig":
        return f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n"
    return f"<entry {i+1}>\nevent: system_poll()\nresult: {body}\n"


def phr(tpl, kind, **kw):
    """Template-dependent phrasings. Semantics identical."""
    P = {
        "orig": {
            "state": "register {r} set to state {v}",
            "num": "register {r} refreshed to {v}",
            "mirror": "register {a} mirrors register {b}",
            "selftest": "register {r} completed a routine self-test",
            "operand": "sensor {r} measured {gv}; the alert threshold is {thr}",
            "verdict": "alert check result for sensor {r}: threshold {v}",
            "neutral": "checkpoint saved for the sensor {r} review",
            "outcome": "the outcome of the sensor {r} alert check was archived for the report",
            "flag": "the alert status flag for sensor {r} was set according to the check "
                    "result",
        },
        "para": {
            "state": "the status of register {r} was logged as {v}",
            "num": "a reading of {v} was recorded for register {r}",
            "mirror": "register {a} tracks register {b}",
            "selftest": "register {r} passed its scheduled self-check",
            "operand": "sensor {r} returned a reading of {gv} against an alert limit of "
                       "{thr}",
            "verdict": "the {r} alert evaluation concluded: limit {v}",
            "neutral": "a checkpoint of the {r} review was written to disk",
            "outcome": "the conclusion of the {r} check was filed in the report",
            "flag": "the alarm flag for {r} was updated in line with the check outcome",
        },
    }
    return P[tpl][kind].format(**kw)


PICK_LAST = os.environ.get("SPRAG_PICK_LAST", "0") == "1"
DIRECT_NTOK = int(os.environ.get("SPRAG_DIRECT_NTOK", "10"))


def pickw(o, words, last=False):
    last = last or PICK_LAST
    u = o.upper()
    f = (lambda w: u.rfind(w)) if last else (lambda w: u.find(w))
    hits = [(f(w), w) for w in words if f(w) >= 0]
    if not hits:
        return None
    return max(hits)[1] if last else min(hits)[1]


def build(tpl, rng, specials, n=N_BODY):
    others = [x for x in NAMES if x not in specials.get("_names", [])]
    body = {}
    for i, txt in specials.items():
        if i != "_names":
            body[i] = txt
    for i in range(n):
        if i not in body:
            body[i] = phr(tpl, "num", r=rng.choice(others), v=rng.randrange(100, 999))
    return [wrap(tpl, i, body[i]) for i in range(n)]


def run_cell(llm, H_ids, H, H_kv, tail, head_full, evs, served, q, ntok, pickf):
    eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in evs]
    spans, cur = [], H
    for e in eids:
        spans.append((cur, cur + len(e)))
        cur += len(e)
    total = cur
    flat = [t for e in eids for t in e]
    full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
    crows = [pp for i in served for pp in range(*spans[i])]
    rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
    sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
    o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), ntok)
    del c, sub, full_kv
    torch.cuda.empty_cache()
    return pickf(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default="./out/fz.json")
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

    for tpl in TPLS:
        for it in range(args.items):
            rec = {"tpl": tpl, "it": it}

            # cell1: binding recall, none-cell (bin2, no state word served)
            D, S, T = rng.sample(NAMES, 3)
            gold = rng.choice(STATUS2)
            m = rng.randrange(8, 16)
            sp = {DEC_IDX: phr(tpl, "selftest", r=D),
                  m: phr(tpl, "state", r=S, v=gold),
                  CAR_IDX: phr(tpl, "mirror", a=T, b=S), "_names": [D, S, T]}
            evs = build(tpl, rng, sp)
            q = (f"\n\nQuestion: What state is register {T} in? "
                 "Answer ONLINE or OFFLINE:")
            rec["c1"] = int(run_cell(llm, H_ids, H, H_kv, tail, head_full, evs,
                                     [DEC_IDX, CAR_IDX], q, DIRECT_NTOK,
                                     lambda o: pickw(o, STATUS2)) == gold)

            # cell2: binding gradient (explicit verdict dropped; 3 carrier wordings)
            D, S = rng.sample(NAMES, 2)
            gv = rng.randrange(100, 999)
            thr = rng.randrange(100, 999)
            while abs(gv - thr) < 30:
                thr = rng.randrange(100, 999)
            vgold = "EXCEEDED" if gv > thr else "NORMAL"
            m = rng.randrange(8, 14)
            for ck in ("neutral", "outcome", "flag"):
                sp = {DEC_IDX: phr(tpl, "selftest", r=D),
                      m: phr(tpl, "operand", r=S, gv=gv, thr=thr),
                      m + 1: phr(tpl, "verdict", r=S, v=vgold),
                      CAR_IDX: phr(tpl, ck, r=S), "_names": [D, S]}
                evs = build(tpl, rng, sp)
                q = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
                     "Answer EXCEEDED or NORMAL:")
                rec[f"c2_{ck}"] = int(run_cell(llm, H_ids, H, H_kv, tail, head_full, evs,
                                               [DEC_IDX, CAR_IDX], q, DIRECT_NTOK,
                                               lambda o: pickw(o, ["EXCEEDED", "NORMAL"]))
                                      == vgold)

            # cell3: dir4 cliff
            D, S, T = rng.sample(NAMES, 3)
            g4 = rng.choice(DIR4)
            dm = rng.choice(DIR4)
            m = rng.randrange(8, 16)
            sp = {DEC_IDX: phr(tpl, "state", r=D, v=dm),
                  m: phr(tpl, "state", r=S, v=g4),
                  CAR_IDX: phr(tpl, "mirror", a=T, b=S), "_names": [D, S, T]}
            evs = build(tpl, rng, sp)
            q = (f"\n\nQuestion: What state is register {T} in? "
                 "Answer NORTH, SOUTH, EAST, or WEST:")
            rec["c3"] = int(run_cell(llm, H_ids, H, H_kv, tail, head_full, evs,
                                     [DEC_IDX, CAR_IDX], q, DIRECT_NTOK,
                                     lambda o: pickw(o, DIR4)) == g4)

            # cell4: chain root/leaf/full (depth 3, trace)
            D, S, M1, M2, T = rng.sample(NAMES, 5)
            v1 = rng.choice(STATUS2)
            dm2 = rng.choice(STATUS2)
            m = rng.randrange(8, 12)
            C1, C2, C3 = 13, 18, 23
            sp = {DEC_IDX: phr(tpl, "state", r=D, v=dm2),
                  m: phr(tpl, "state", r=S, v=v1),
                  C1: phr(tpl, "mirror", a=M1, b=S),
                  C2: phr(tpl, "mirror", a=M2, b=M1),
                  C3: phr(tpl, "mirror", a=T, b=M2), "_names": [D, S, M1, M2, T]}
            evs = build(tpl, rng, sp)
            A = " Answer ONLINE or OFFLINE." + TRACE
            for key, served, reg in (("c4_root", [DEC_IDX, C1], M1),
                                     ("c4_leaf", [DEC_IDX, C3], T),
                                     ("c4_full", [DEC_IDX, C1, C2, C3], T)):
                q = f"\n\nQuestion: What state is register {reg} in?{A}"
                rec[key] = int(run_cell(llm, H_ids, H, H_kv, tail, head_full, evs,
                                        served, q, 160,
                                        lambda o: pickw(o, STATUS2, last=True)) == v1)
            rows.append(rec)
            print(f"[fz] {tpl} it{it} | c1={rec['c1']} "
                  f"c2={rec['c2_neutral']}{rec['c2_outcome']}{rec['c2_flag']} "
                  f"c3={rec['c3']} c4={rec['c4_root']}{rec['c4_leaf']}{rec['c4_full']}",
                  flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("\n===== FROZEN REPLICATION (fresh seeds; orig vs paraphrase) =====")
    for tpl in TPLS:
        g = [r for r in rows if r["tpl"] == tpl]
        n = len(g)

        def mn(k):
            return sum(r[k] for r in g) / n
        print(f"{tpl:5s} c1_recall={mn('c1'):.2f} | grad n/o/f="
              f"{mn('c2_neutral'):.2f}/{mn('c2_outcome'):.2f}/{mn('c2_flag'):.2f} | "
              f"dir4={mn('c3'):.2f} | chain r/l/f="
              f"{mn('c4_root'):.2f}/{mn('c4_leaf'):.2f}/{mn('c4_full'):.2f}")
    print("FZ_DONE", flush=True)


if __name__ == "__main__":
    main()
