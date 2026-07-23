"""kv_poscontrol.py -- X11: position-CONTROLLED presence x root-donor factorial.

WHY. cz2 (kv_causal2) compared the root-donor contrast with the downstream mention
co-served (cells a_{v}_{j}) against a "mention-absent" cell (a_{v}_NO). Those two
protocols differ in serve-set membership AND in the appended query's absolute position:
the query is placed at max(served position)+1, and the mention row sits at DW_IDX=18
while the root sits at C1_IDX=13. So the observed crossover (root effect .398 -> .083 on
Qwen3, .091 -> .220 on Gemma-4) confounds "the mention is present" with "the query moved
5 event-slots further from the root". This run removes that confound.

DESIGN. Same 24-event log as cz2. Three serve conditions per root donor v:

  men : {DEC, C1@v} + {DW@j}, DW text = "register M1 was listed in the scheduled
        audit inventory for this cycle"     (mentions the QUERIED target M1)
  fil : {DEC, C1@v} + {DW@j}, DW text = same template with an UNRELATED register U
        (referent swap: identical template/slot/length, does not mention M1)
  abs : {DEC, C1@v} only                    (cz2's original mention-absent cell)

men and fil serve a row at the SAME slot, so the query lands at the SAME absolute
position; they differ ONLY in whether the co-served row refers to the queried target.
U is chosen so the DW event tokenizes to EXACTLY the same length as with M1 (searched
over candidate names; dw_len_men/dw_len_fil are recorded so equality is auditable).
DEC and C1 sit at slots 0 and 13, before DW at 18, so under causal attention their rows
are bit-identical across the men/fil encodings -- the abs arm is therefore well defined.

DECOMPOSITION (per checkpoint, on the same complete-case items):
  root(men) - root(fil)  = mention effect, POSITION-CONTROLLED  <- the clean estimand
  root(fil) - root(abs)  = pure position/row-count effect
  root(men) - root(abs)  = cz2's original, confounded contrast (= sum of the two)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_poscontrol \
        --items 128 --seed 8200 --out ./out/pc.json
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
from kvmemory.kv_causal2 import (STATUS2, N_BODY, DEC_IDX, SRC_LO, SRC_HI, C1_IDX,
                                 DW_IDX, TRACE, DW_STYLES, pick_last, encode_variant,
                                 sub_rows)


def ev(i, txt):
    return f"<step {i+1}>\naction: reg_op()\nobservation: {txt}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8200)
    ap.add_argument("--dw", default="neutral", choices=sorted(DW_STYLES))
    ap.add_argument("--menu", action="store_true",
                    help="menu/direct readout (Gemma-4's free-generation is unreliable)")
    ap.add_argument("--gen", type=int, default=160)
    ap.add_argument("--out", default="./out/pc.json")
    args = ap.parse_args()
    dw_txt = DW_STYLES[args.dw]

    llm = HFBackend()
    print("MODEL_CLASS", type(llm.model).__name__, flush=True)
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)
    tlen = lambda t: len(llm.tok(t, add_special_tokens=False).input_ids)
    rows = []

    for it in range(args.items):
        D, S, M1 = rng.sample(NAMES, 3)
        dm = rng.choice(STATUS2)
        m = rng.randrange(SRC_LO, SRC_HI)
        others = [n for n in NAMES if n not in (D, S, M1)]

        men_txt = dw_txt.format(m=M1)
        L_men = tlen(ev(DW_IDX, men_txt))
        # referent swap: same template, unrelated register, EXACT same token length
        cands = [u for u in others if tlen(ev(DW_IDX, dw_txt.format(m=u))) == L_men]
        U = rng.choice(cands) if cands else rng.choice(others)
        fil_txt = dw_txt.format(m=U)
        L_fil = tlen(ev(DW_IDX, fil_txt))

        fill = {i: f"register {rng.choice([n for n in others if n != U])} "
                   f"refreshed to {rng.randrange(100, 999)}" for i in range(N_BODY)}
        src_budget = max(tlen(ev(m, f"register {S} set to state {v}")) for v in STATUS2)
        dw_budget = max(L_men, L_fil)

        enc = {}
        for v in STATUS2:
            for cont, txt in (("men", men_txt), ("fil", fil_txt)):
                body = dict(fill)
                body[DEC_IDX] = f"register {D} set to state {dm}"
                body[m] = f"register {S} set to state {v}"
                body[C1_IDX] = f"register {M1} mirrors register {S}"
                body[DW_IDX] = txt
                evs = [ev(i, body[i]) for i in range(N_BODY)]
                enc[(v, cont)] = encode_variant(llm, H_ids, H, evs,
                                                {m: src_budget, DW_IDX: dw_budget})

        if args.menu:
            q = (f"\n\nQuestion: What state is register {M1} in? "
                 "Answer with the single state word only (ONLINE or OFFLINE):")
        else:
            q = f"\n\nQuestion: What state is register {M1} in? Answer ONLINE or OFFLINE." + TRACE
        qids = llm._ids(q + tail)

        rec = {"it": it, "dm": dm, "U": U, "M1": M1,
               "dw_len_men": L_men, "dw_len_fil": L_fil, "len_match": int(L_men == L_fil)}

        for v in STATUS2:
            # --- abs: cz2's original mention-absent cell (query lands after C1) ---
            kv_i, rm_i, sp_i, ei_i = enc[(v, "men")]
            sub_i, cr_i = sub_rows(kv_i, rm_i, sp_i, ei_i, [DEC_IDX, C1_IDX])
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub_i, cr_i)])
            o, _, _ = llm._greedy_pos(c, p, qids, args.gen)
            rec[f"a_{v[:2]}_abs"] = pick_last(o)
            rec[f"qpos_{v[:2]}_abs"] = int(max(cr_i)) + 1
            del c

            # --- men / fil: position-matched, differ only in the DW referent ---
            for cont in ("men", "fil"):
                kv_i, rm_i, sp_i, ei_i = enc[(v, cont)]
                sub_i, cr_i = sub_rows(kv_i, rm_i, sp_i, ei_i, [DEC_IDX, C1_IDX])
                for j in STATUS2:
                    kv_j, rm_j, sp_j, ei_j = enc[(j, cont)]
                    sub_j, cr_j = sub_rows(kv_j, rm_j, sp_j, ei_j, [DW_IDX])
                    c, p = assemble(llm, [(H_kv, list(range(H))),
                                          (sub_i, cr_i), (sub_j, cr_j)])
                    o, _, _ = llm._greedy_pos(c, p, qids, args.gen)
                    rec[f"a_{v[:2]}_{j[:2]}_{cont}"] = pick_last(o)
                    rec[f"qpos_{v[:2]}_{j[:2]}_{cont}"] = int(max(cr_j)) + 1
                    del c, sub_j
            del sub_i
        del enc
        torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[pc] it{it} lenmatch={rec['len_match']} "
              f"abs={rec['a_ON_abs']}/{rec['a_OF_abs']} "
              f"men={rec['a_ON_ON_men']}/{rec['a_OF_ON_men']} "
              f"fil={rec['a_ON_ON_fil']}/{rec['a_OF_ON_fil']}", flush=True)

    json.dump({"rows": rows, "dw": args.dw, "menu": bool(args.menu)},
              open(args.out, "w"), indent=1)

    # ---- summary: root effect on P(ONLINE) under each protocol ----
    def on(x):
        return 1 if x == "ONLINE" else (0 if x == "OFFLINE" else None)

    def root_eff(cells_on, cells_of):
        vals = []
        for r in rows:
            a = [on(r.get(k)) for k in cells_on]
            b = [on(r.get(k)) for k in cells_of]
            if any(x is None for x in a + b):
                continue
            vals.append(sum(a) / len(a) - sum(b) / len(b))
        n = len(vals)
        return (sum(vals) / n if n else float("nan")), n

    print(f"\n===== X11 position-controlled presence x root-donor (n={len(rows)}) =====")
    qp_ok = all(r[f"qpos_{v}_{j}_men"] == r[f"qpos_{v}_{j}_fil"]
                for r in rows for v in ("ON", "OF") for j in ("ON", "OF"))
    print(f"query-position identical men vs fil: {qp_ok}   "
          f"len-match items: {sum(r['len_match'] for r in rows)}/{len(rows)}")
    for tag, con, cof in (
            ("men", ["a_ON_ON_men", "a_ON_OF_men"], ["a_OF_ON_men", "a_OF_OF_men"]),
            ("fil", ["a_ON_ON_fil", "a_ON_OF_fil"], ["a_OF_ON_fil", "a_OF_OF_fil"]),
            ("abs", ["a_ON_abs"], ["a_OF_abs"])):
        e, n = root_eff(con, cof)
        print(f"  root effect | {tag} = {e:+.3f}  (n={n})")
    print("PC_DONE", flush=True)


if __name__ == "__main__":
    main()
