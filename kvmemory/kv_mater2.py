"""kv_mater2.py -- X9 (verdict-5 capstone): ACTIVE MATERIALIZATION across carrier classes.

Verdict-5's one recommended experiment: can an agent-emitted event whose TEXT contains
neither the answer nor the operands cause prefill to commit a derived 1-bit conclusion
(threshold verdict) into that event's KV rows -- so that after the source is dropped from
the serving set the conclusion is still behaviorally recoverable? kv_noteknob2 showed one
NL compute-note lifts harvest-drop .54 -> .70 on this domain; X9 turns that into the
carrier-CLASS comparison, donor-paired so any source-free reading of the served material
is a structural floor.

Item (N_BODY=24, verdict domain from kv_noteknob2):
  slot 0   DEC  "sensor {D} logged a routine idle reading"      served, donor-invariant
  slot 2   UREG "register {U} set to state {us}"                served, donor-invariant;
                                                                unrelated-query echo control
  slot m   SRC  "sensor {S} measured {gv}; the alert threshold is {thr}"   NEVER served;
                donor pair = gv_hi (EXCEEDED) vs gv_lo (NORMAL), thr fixed; slot position-
                budgeted so all later rows sit at identical positions across donors
  slot m+1 CARRIER (manipulated arm; pas/bind/dir/nld byte-identical across donors, assert)
  slot 23  DST  "the operator reviewed sensor {S} against the alert threshold"  served

Arms (2 templates each; item i uses template i%2; leak-asserted for non-exp arms):
  pas   passive mention (result archived; no target, no directive)
  bind  grounded binding (register T set per the verdict; value-free)
  dir   structured materialization directive <MATERIALIZE .../> (no answer, no operands)
  nld   NL compute directive (kv_noteknob2's note = replication bridge)
  exp   explicit text (verdict written in carrier; upper bound, donor-variant)

Readouts (12-tok greedy):
  harv = assemble [H; rows{DEC,UREG,CARRIER,DST}] from each donor's full-log cache;
         Qmain (+Qtgt for T-arms, +Qunrel echo control)
  sel  = fresh text prompt of the same served events (isolated floor; identical text
         across donors for non-exp arms => structurally donor-blind)

Primary: donor-transition class on Qmain/harv per arm (follow = each donor answers its own
gold; anti; const; other). Pre-registered: follow(dir) - follow(pas) CI excl. 0 on both
models; dir vs bind TOST +/-.075; Qunrel const-dominant (no global echo).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_mater2 --items 2 --seed 999 --out ./out/x9.json
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
from kvmemory.kv_causal2 import pick_last, encode_variant, sub_rows

STATUS2 = ["ONLINE", "OFFLINE"]
N_BODY = 24
DEC_IDX, U_IDX = 0, 2
DST_IDX = N_BODY - 1

ARMS = {
    "pas": ["note: the threshold check for sensor {S} has completed and its result was "
            "archived",
            "log entry: the outcome of the {S} threshold comparison was filed in the "
            "maintenance record"],
    "bind": ["status register {T} is now set according to the threshold verdict for "
             "sensor {S}",
             "register {T} now holds the outcome of the threshold check for sensor {S}"],
    "dir": ['<MATERIALIZE target="{T}" relation="threshold_verdict" subject="{S}"/>',
            '<MATERIALIZE subject="{S}" relation="threshold_verdict" target="{T}"/>'],
    "nld": ["audit note: determine the threshold verdict for sensor {S} now and record it "
            "here for the final report",
            "reminder: work out whether sensor {S} went above its alert threshold and note "
            "the verdict here"],
    "exp": ["status register {T} recorded the threshold verdict for sensor {S}: {V}",
            "verdict for sensor {S} filed in register {T}: {V}"],
}
T_ARMS = ("bind", "dir", "exp")
DONORS = (("hi", "EXCEEDED"), ("lo", "NORMAL"))


def pickv(o):
    u = o.upper()
    a, b = u.rfind("EXCEEDED"), u.rfind("NORMAL")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "NORMAL"
    if b < 0:
        return "EXCEEDED"
    return "EXCEEDED" if a > b else "NORMAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1100)
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--out", default="./out/x9.json")
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

    for it in range(args.items):
        tpl = it % 2
        D, S, T, U = rng.sample(NAMES, 4)
        us = rng.choice(STATUS2)
        thr = rng.randrange(200, 800)
        gv = {"hi": rng.randrange(thr + 30, 999), "lo": rng.randrange(100, thr - 29)}
        m = rng.randrange(N_BODY // 3, (2 * N_BODY) // 3)
        others = [n for n in NAMES if n not in (D, S, T, U)]
        fill = {i: f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
                for i in range(N_BODY)}
        fill[DEC_IDX] = f"sensor {D} logged a routine idle reading"
        fill[U_IDX] = f"register {U} set to state {us}"
        fill[DST_IDX] = f"the operator reviewed sensor {S} against the alert threshold"
        qmain = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
                 "Answer EXCEEDED or NORMAL:")
        qtgt = (f"\n\nQuestion: What verdict is recorded for status register {T}? "
                "Answer EXCEEDED or NORMAL:")
        qunrel = (f"\n\nQuestion: What state is register {U} in? "
                  "Answer with the single state word only:")
        src_txts = {d: f"sensor {S} measured {gv[d]}; the alert threshold is {thr}"
                    for d, _ in DONORS}
        src_budget = max(len(llm.tok(f"<step {m+1}>\naction: reg_op()\nobservation: "
                                     f"{src_txts[d]}\n", add_special_tokens=False).input_ids)
                         for d, _ in DONORS)
        rec = {"it": it, "tpl": tpl, "m": m, "S": S, "T": T, "U": U, "us": us,
               "thr": thr, "gv_hi": gv["hi"], "gv_lo": gv["lo"]}
        for arm, tpls in ARMS.items():
            cars = {d: tpls[tpl].format(S=S, T=T, V=gold) for d, gold in DONORS}
            if arm != "exp":
                assert cars["hi"] == cars["lo"], f"carrier differs across donors: {arm}"
                up = cars["hi"].upper()
                assert "EXCEEDED" not in up and "NORMAL" not in up, f"answer leak: {arm}"
                assert not any(str(x) in cars["hi"] for x in (gv["hi"], gv["lo"], thr)), \
                    f"operand leak: {arm}"
            car_budget = max(len(llm.tok(f"<step {m+2}>\naction: reg_op()\nobservation: "
                                         f"{cars[d]}\n", add_special_tokens=False).input_ids)
                             for d, _ in DONORS)
            served = [DEC_IDX, U_IDX, m + 1, DST_IDX]
            for d, gold in DONORS:
                body = dict(fill)
                body[m] = src_txts[d]
                body[m + 1] = cars[d]
                evs = [f"<step {i+1}>\naction: reg_op()\nobservation: {body[i]}\n"
                       for i in range(N_BODY)]
                kv, rm, sp, ei = encode_variant(llm, H_ids, H, evs,
                                                {m: src_budget, m + 1: car_budget})
                sub, cr = sub_rows(kv, rm, sp, ei, served)
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, cr)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(qmain + tail), args.gen)
                rec[f"{arm}_h_{d}"] = pickv(o)
                rec[f"{arm}_hraw_{d}"] = o[:100]
                del c
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, cr)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(qunrel + tail), args.gen)
                rec[f"{arm}_u_{d}"] = pick_last(o)
                del c
                if arm in T_ARMS:
                    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, cr)])
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(qtgt + tail), args.gen)
                    rec[f"{arm}_t_{d}"] = pickv(o)
                    del c
                if arm == "exp" or d == "hi":
                    sel_txt = "".join(evs[i] for i in served)
                    o, _, _ = llm._greedy(DynamicCache(), 0,
                                          llm._ids(head_full + sel_txt + qmain + tail), args.gen)
                    rec[f"{arm}_s_{d}"] = pickv(o)
                del kv, sub
                torch.cuda.empty_cache()
        rows.append(rec)
        fl = {a: int(rec[f"{a}_h_hi"] == "EXCEEDED" and rec[f"{a}_h_lo"] == "NORMAL")
              for a in ARMS}
        print(f"[x9] it{it} tpl{tpl} follow=" +
              " ".join(f"{a}:{fl[a]}" for a in ARMS) + " | unrel_flip=" +
              " ".join(f"{a}:{int(rec[f'{a}_u_hi'] != rec[f'{a}_u_lo'])}" for a in ARMS),
              flush=True)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    print("\n===== X9 ACTIVE MATERIALIZATION (donor transition on Qmain / harv) =====")
    for a in ARMS:
        fo = sum(1 for r in rows
                 if r[f"{a}_h_hi"] == "EXCEEDED" and r[f"{a}_h_lo"] == "NORMAL")
        an = sum(1 for r in rows
                 if r[f"{a}_h_hi"] == "NORMAL" and r[f"{a}_h_lo"] == "EXCEEDED")
        co = sum(1 for r in rows
                 if r[f"{a}_h_hi"] == r[f"{a}_h_lo"] and r[f"{a}_h_hi"] is not None)
        ot = n - fo - an - co
        uf = sum(1 for r in rows if r[f"{a}_u_hi"] != r[f"{a}_u_lo"])
        ua = sum(1 for r in rows for d in ("hi", "lo")
                 if r[f"{a}_u_{d}"] == r["us"]) / (2 * n)
        line = (f"{a:4s} n={n} follow={fo} anti={an} const={co} other={ot} "
                f"| unrel flip={uf} acc={ua:.2f}")
        if a == "exp":
            sf = sum(1 for r in rows
                     if r["exp_s_hi"] == "EXCEEDED" and r["exp_s_lo"] == "NORMAL")
            line += f" | sel_follow={sf}"
        else:
            se = sum(1 for r in rows if r[f"{a}_s_hi"] == "EXCEEDED")
            line += f" | sel EXC={se}/{n}"
        print(line)
    for tp in (0, 1):
        g = [r for r in rows if r["tpl"] == tp]
        if g:
            print(f"tpl{tp} n={len(g)} follow: " + " ".join(
                f"{a}:{sum(1 for r in g if r[f'{a}_h_hi'] == 'EXCEEDED' and r[f'{a}_h_lo'] == 'NORMAL')}"
                for a in ARMS))
    for a in T_ARMS:
        tf = sum(1 for r in rows
                 if r[f"{a}_t_hi"] == "EXCEEDED" and r[f"{a}_t_lo"] == "NORMAL")
        print(f"qtgt {a}: follow={tf}/{n}")
    print("X9_DONE", flush=True)


if __name__ == "__main__":
    main()
