"""kv_harvest4.py -- X8b: the adjudication package for the real-dialogue negative (verdict-4).

Two components, frozen:

A. RECOGNITION readout on the SAME natural carriers as kv_harvest3 (win=0 selection):
   each QA becomes a two-choice task gold-vs-hard-negative; hard negative = the gold of the
   nearest other QA in the SAME conversation with a different answer (same category
   preferred; deterministic qid order). Menu order balanced by qid hash; ALL arms see the
   identical menu. Arms: none / text / iso / harv / goldev. Addresses "written but not
   readable under free generation" (own §4: recognition can reveal what recall cannot).

B. PATH POSITIVE CONTROL (donor-paired injected carrier): subset of QAs whose gold appears
   verbatim in >= 1 evidence turn (<=4 per conversation, strong-first, deterministic).
   Two conversation versions: original vs counterfactual (gold -> hard negative substring
   swap inside evidence turns). One synthetic carrier turn appended at the END referencing
   the question but NOT the value. Serve ONLY the injected turn's rows (header + carrier):
     harv_orig  rows harvested from the original-version full prefill
     harv_cf    rows harvested from the counterfactual-version full prefill
     iso_inj    the injected turn encoded alone at the original positions
   follow-donor = pick gold under orig AND pick neg under cf. If follow >> iso/chance the
   encode->harvest->splice->recognition path is demonstrated ON REAL DIALOGUE, making the
   natural-carrier result interpretable; if the path itself fails, the §6.2 negative stays
   protocol-scoped (as已降格).

Pre-registered: natural harv - iso equivalence bound +/-.05 (recognition), conversation-
clustered on LOCOMO (REALTALK descriptive; 3 participant components). Readout: 8-token
greedy, parse first standalone A/B.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_harvest4 --ds realtalk --shard 0 \
        --nshards 10 --max_qa 2 --out ./out/h4.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_harvest3 import load_realtalk, load_locomo, select_qa

N_INJECT = 4
AB_RE = re.compile(r"\b([AB])\b")


def pick_ab(o):
    m = AB_RE.search(o.upper())
    return m.group(1) if m else None


def recog_q(question, opt_a, opt_b):
    return (f"\n\nQuestion: {question}\n"
            f"Option A: {opt_a}\nOption B: {opt_b}\n"
            "Which option is correct? Answer A or B.")


def hard_neg(qas, qa):
    g = qa["gold"].strip().lower()
    cands = [q2 for q2 in qas if q2["gold"].strip().lower() != g]
    if not cands:
        return None
    same = [q2 for q2 in cands if q2["cat"] == qa["cat"]]
    pool = same if same else cands
    pool.sort(key=lambda q2: q2["q"])
    return pool[0]["gold"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo", "all"], required=True)
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--max_qa", type=int, default=0)
    ap.add_argument("--queue_dir", default="", help="shared atomic queue over (ds, conv) "
                    "tasks; overrides --shard/--nshards")
    ap.add_argument("--out", default="./out/h4.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    if args.ds == "all":
        tasks = ([("realtalk", c) for c in load_realtalk(args.rt_path)]
                 + [("locomo", c) for c in load_locomo(args.lc_path)])
    else:
        loader = load_realtalk(args.rt_path) if args.ds == "realtalk" else load_locomo(args.lc_path)
        tasks = [(args.ds, c) for c in loader]
    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
    else:
        tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    rows, inj_rows = [], []

    def prefill(turn_list):
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids)
                for _, _, t in turn_list]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)
        return kv, spans, eids

    def ask_rows(full_kv, spans, idxs, qids_t):
        crows = [pp for i in idxs for pp in range(*spans[i])]
        rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
        sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
        c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
        o, _, _ = llm._greedy_pos(c, p, qids_t, 8)
        del c, sub
        return pick_ab(o)

    def ask_text(body, qids_txt_q):
        from transformers import DynamicCache
        o, _, _ = llm._greedy(DynamicCache(), 0,
                              llm._ids(head + body + qids_txt_q), 8)
        return pick_ab(o)

    for ds_name, cv in tasks:
        if args.queue_dir:
            try:
                os.mkdir(os.path.join(args.queue_dir, f"{ds_name}_{cv['cid']}"))
            except (FileExistsError, OSError):
                continue
        qas = select_qa(cv)
        if args.max_qa:
            qas = qas[: args.max_qa]
        if not qas:
            continue
        turns = cv["turns"]
        full_kv, spans, eids = prefill(turns)
        print(f"[h4] {cv['cid']} turns={len(turns)} qa={len(qas)}", flush=True)

        # ---------- A. natural carriers, recognition readout ----------
        for qa in qas:
            neg = hard_neg(qas, qa)
            if neg is None:
                continue
            gold = qa["gold"].strip()
            gold_is_a = int(hashlib.sha1(qa["q"].encode()).hexdigest(), 16) % 2 == 0
            a, b = (gold, neg) if gold_is_a else (neg, gold)
            gold_letter = "A" if gold_is_a else "B"
            rq = recog_q(qa["q"], a, b)
            qids_t = llm._ids(rq + tail)
            rec = {"ds": ds_name, "conv": cv["cid"], "cat": qa["cat"],
                   "strong": qa["strong"], "gold": gold, "neg": neg,
                   "gold_letter": gold_letter}
            car_txt = "".join(turns[i][2] + "\n" for i in qa["carriers"])
            gev_txt = "".join(turns[i][2] + "\n" for i in qa["goldev"])
            rec["r_none"] = int(ask_text("", rq + tail) == gold_letter)
            rec["r_text"] = int(ask_text(car_txt, rq + tail) == gold_letter)
            rec["r_goldev"] = int(ask_text(gev_txt, rq + tail) == gold_letter)
            rec["r_harv"] = int(ask_rows(full_kv, spans, qa["carriers"], qids_t)
                                == gold_letter)
            iso_parts = []
            for i in qa["carriers"]:
                aa, bb = spans[i]
                kv_i = encode_block(llm, eids[i], list(range(aa, bb)))
                iso_parts.append((kv_i, list(range(aa, bb))))
            crows = [pp for kv_i, ppl in iso_parts for pp in ppl]
            c, p = assemble(llm, [(H_kv, list(range(H)))] + iso_parts)
            o, _, _ = llm._greedy_pos(c, p, qids_t, 8)
            rec["r_iso"] = int(pick_ab(o) == gold_letter)
            del c, iso_parts
            torch.cuda.empty_cache()
            rows.append(rec)
        print(f"[h4] {cv['cid']} phaseA done n={sum(1 for r in rows if r['conv']==cv['cid'])}",
              flush=True)

        # ---------- B. donor-paired injected carrier (path positive control) ----------
        probe = []
        for qa in qas:
            gl = qa["gold"].strip().lower()
            if len(gl) < 3:
                continue
            if any(gl in turns[i][2].lower() for i in qa["goldev"]):
                neg = hard_neg(qas, qa)
                if neg and neg.strip().lower() not in gl and gl not in neg.strip().lower():
                    probe.append((qa, neg))
        probe.sort(key=lambda x: (-x[0]["strong"], x[0]["q"]))
        probe = probe[:N_INJECT]
        del full_kv
        torch.cuda.empty_cache()

        for qa, neg in probe:
            gold = qa["gold"].strip()
            last_sn = turns[-1][0]
            inj_txt = (f"[session {last_sn+1} follow-up] moderator: for the record, the "
                       f"answer to \"{qa['q']}\" was determined earlier in this chat and "
                       "noted accordingly")
            pat = re.compile(re.escape(gold), re.I)
            variants = {}
            for tag in ("orig", "cf"):
                tl = []
                for j, (sn, d, txt) in enumerate(turns):
                    if tag == "cf" and j in qa["goldev"]:
                        txt = pat.sub(neg, txt)
                    tl.append((sn, d, txt))
                tl.append((last_sn + 1, "INJ", inj_txt))
                variants[tag] = tl
            gold_is_a = int(hashlib.sha1(("inj" + qa["q"]).encode()).hexdigest(), 16) % 2 == 0
            a, b = (gold, neg) if gold_is_a else (neg, gold)
            gold_letter = "A" if gold_is_a else "B"
            neg_letter = "B" if gold_is_a else "A"
            rq = recog_q(qa["q"], a, b)
            qids_t = llm._ids(rq + tail)
            irec = {"ds": ds_name, "conv": cv["cid"], "gold": gold, "neg": neg,
                    "strong": qa["strong"], "gold_letter": gold_letter}
            for tag in ("orig", "cf"):
                kv_v, spans_v, eids_v = prefill(variants[tag])
                inj_idx = len(variants[tag]) - 1
                irec[f"pick_{tag}"] = ask_rows(kv_v, spans_v, [inj_idx], qids_t)
                if tag == "orig":
                    aa, bb = spans_v[inj_idx]
                    kv_i = encode_block(llm, eids_v[inj_idx], list(range(aa, bb)))
                    c, p = assemble(llm, [(H_kv, list(range(H))),
                                          (kv_i, list(range(aa, bb)))])
                    o, _, _ = llm._greedy_pos(c, p, qids_t, 8)
                    irec["pick_iso"] = pick_ab(o)
                    del c, kv_i
                del kv_v
                torch.cuda.empty_cache()
            irec["follow"] = int(irec["pick_orig"] == gold_letter
                                 and irec["pick_cf"] == neg_letter)
            irec["anti"] = int(irec["pick_orig"] == neg_letter
                               and irec["pick_cf"] == gold_letter)
            inj_rows.append(irec)
            print(f"[h4inj] {cv['cid']} orig={irec['pick_orig']} cf={irec['pick_cf']} "
                  f"iso={irec['pick_iso']} follow={irec['follow']} gold={gold[:24]}",
                  flush=True)

    json.dump({"rows": rows, "inj": inj_rows}, open(args.out, "w"), indent=1)
    n = len(rows)
    if n:
        print(f"\n===== {args.ds} recognition arms (n={n}) =====")  # ds may be 'all' 
        for arm in ("none", "text", "iso", "harv", "goldev"):
            print(f"  {arm:7s} {sum(r[f'r_{arm}'] for r in rows)/n:.3f}")
    if inj_rows:
        f = sum(r["follow"] for r in inj_rows)
        av = sum(r["anti"] for r in inj_rows)
        print(f"inject: n={len(inj_rows)} follow={f} anti={av} "
              f"iso_gold={sum(1 for r in inj_rows if r['pick_iso']==r['gold_letter'])}")
    print("H4_DONE", flush=True)


if __name__ == "__main__":
    main()
