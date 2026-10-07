"""kv_e2swap.py -- the bridge experiment: is the narrow form's +1.87 log-odds EVIDENCE-driven?

The natural-occurrence serve sets (kv_e2natsets; paper App. I) found that when a served row states the answer, that
row's in-situ KV (`bhot`) reads +1.87 log-odds more confidently than the same text re-encoded
(`self`), placebo flat.  Two readings are compatible with that number:

    evidence-driven   the in-situ KV is a more faithful representation of what the row SAYS,
                      so the model follows the row more strongly -- whichever value it states
    prior-driven      the in-situ KV somehow boosts the GOLD regardless of what the row states

They separate under a counterfactual: rewrite the value v -> v' in the conversation (R3: every
word-boundary occurrence, residual asserted 0), re-prefill, and read the same served rows again.

    o_*   the original world          c_*   the counterfactual world (same rows, same positions
                                             up to token-length drift of v')
    lo    log p(gold) - log p(neg)    -- so in the cf world the EVIDENCE-consistent answer is neg
                                         and lo should go NEGATIVE (built-in positive control)

    Delta_KV   = o_bhot - c_bhot        Delta_text = o_self - c_self
    I          = Delta_KV - Delta_text  = (bhot - self)_orig - (bhot - self)_cf

    evidence-driven  =>  (bhot - self)_cf ~ -1.87 (mirror image)  =>  I ~ +3.7
    prior-driven     =>  (bhot - self)_cf ~ +1.87                  =>  I ~ 0

Only items whose SERVED rows contain the gold verbatim (word boundary, case-insensitive) are
testable -- a paraphrased value row cannot be rewritten by string and would leak the original
value into the cf world.  Everything else (menu, arms, hot window, V-in-hot flags) is
kv_e2hot2's code path, so the o_* numbers at hot=0 must reproduce nat_sh*.log bit-for-bit.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa SPRAG_ENCODE_CHUNK=4096 \\
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e2swap \\
        --sets ./out/e2_natsets.json --negs ./out/e2_negs.json \\
        --cells 5:0,5:4,5:16 --out ./out/e2swap.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest4 import recog_q
from kvmemory.kv_e3negs import qkey
from kvmemory.locomo import load_locomo
from kvmemory.rA_roles import _utterance
from kvmemory.kv_e2hot2 import prefill_fresh, log_odds, INSTR

ARMS = ("bhot", "self", "harv")


def gold_pattern(gold):
    """kv_e3ctrl's word-boundary rule, verbatim: a short gold must not rewrite the inside of
    longer words ('Rob' inside 'problems')."""
    pb = re.escape(gold)
    if gold[:1].isalnum():
        pb = r"\b" + pb
    if gold[-1:].isalnum():
        pb = pb + r"\b"
    return re.compile(pb, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--negs", required=True)
    ap.add_argument("--labels", default="./out/rA_labels.json")
    ap.add_argument("--loose", default="./out/rA_loose.json")
    ap.add_argument("--cells", default="5:0,5:4,5:16")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--done_from", default="")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_nv", type=int, default=1,
                    help="only items with >= this many value rows served (natural sets)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    CELLS = []
    for tok in args.cells.split(","):
        tok = tok.strip()
        if tok:
            k, _, h = tok.partition(":")
            CELLS.append((int(k), int(h or 0)))
    assert CELLS and any(h == 0 for _, h in CELLS), "need a hot=0 cell as the gate"
    HOTS = sorted({h for _, h in CELLS})

    VSET = {}
    for r in json.load(open(args.labels))["records"]:
        VSET[(str(r["conv"]), int(r["qi"]))] = set(r["V"]["N1"])
    for r in json.load(open(args.loose))["records"]:
        key = (str(r["conv"]), int(r["qi"]))
        if key in VSET:
            VSET[key] |= set(r.get("V_loose") or [])

    DONE = set()
    if args.done_from:
        import glob as _glob
        pat = re.compile(r"^\[e2swap\] (\S+) q(\d+) \|")
        for g in args.done_from.split(":"):
            for f in _glob.glob(g):
                for line in open(f, errors="ignore"):
                    m = pat.match(line)
                    if m:
                        DONE.add((m.group(1), int(m.group(2))))
        print("[e2swap] resume: %d items already measured" % len(DONE), flush=True)

    S = json.load(open(args.sets))
    items = [it for it in S["items"] if it.get("nV_served", 0) >= args.min_nv]
    print("[e2swap] %d items with nV_served >= %d" % (len(items), args.min_nv), flush=True)
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    HPOS = list(range(H))
    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    NEGS = json.load(open(args.negs))["negs"]

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        return cand.get("A", 0.0), cand.get("B", 0.0)

    def world(turn_texts):
        """token ids / spans / flat stream for one text world; positions start after the header."""
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids) for t in turn_texts]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        return eids, spans, [t for e in eids for t in e], cur

    def rp(full_kv, eids, spans, units, mode, hot_u, menu):
        hotset = set(hot_u)
        fresh = None
        if mode == "bhot":
            served = [i for i in units if i not in hotset]
            blocks = [(H_kv, HPOS)]
            if served:
                crows = [pp for i in served for pp in range(*spans[i])]
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                blocks.append(([(K.index_select(2, rt), V.index_select(2, rt))
                                for K, V in full_kv], crows))
            hu = [i for i in units if i in hotset]
            if hu:
                fresh = ([t for i in hu for t in eids[i]],
                         [p for i in hu for p in range(*spans[i])])
        elif mode == "self":
            crows = [pp for i in units for pp in range(*spans[i])]
            blocks = [(H_kv, HPOS),
                      (encode_block(llm, [t for i in units for t in eids[i]], crows), crows)]
        else:  # harv
            crows = [pp for i in units for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            blocks = [(H_kv, HPOS),
                      ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv],
                       crows)]
        c, pp = assemble(llm, blocks)
        if fresh is not None:
            hot_ids, hot_pos = fresh
            c = prefill_fresh(llm, c, pp.shape[0], hot_ids, hot_pos)
            pp = torch.cat([pp, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
        pa, pb = probe(c, pp, menu[0])
        del c
        tot = pa + pb
        return (round((pa if menu[1] else pb) / tot if tot > 0 else 0.5, 4),
                round(log_odds(pa, pb, menu[1]), 4))

    by_conv = defaultdict(list)
    for it in items:
        by_conv[it["conv"]].append(it)
    convs = [c for i, c in enumerate(sorted(by_conv)) if i % args.nshards == args.shard]
    print("[e2swap] cells (k, hot) = %s" % CELLS, flush=True)
    rows, skipped = [], defaultdict(int)
    for cid in convs:
        ep = eps[cid]
        turns = [_utterance(s) for s in ep.segments]
        NT = len(turns)
        for _it in by_conv[cid]:
            assert _it["n_turns"] == NT, "n_turns mismatch conv %s" % cid
        HOT = {h: list(range(max(0, NT - h), NT)) for h in HOTS}
        eids, spans, flat, cur = world(turns)
        print(f"[e2swap] conv {cid} turns={NT} tok={cur} items={len(by_conv[cid])}", flush=True)
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)

        for it in by_conv[cid]:
            if (cid, int(it["qi"])) in DONE:
                continue
            qt = f"\n\nQuestion: {it['question']}\n{INSTR}" + tail
            gold = it["gold"]
            pat = gold_pattern(gold)
            srv, pl = sorted(it["serve"]), sorted(it["placebo"])

            # ---- testable only if a SERVED row states the gold verbatim ----
            srv_hit = [i for i in srv if pat.search(turns[i])]
            if not srv_hit:
                skipped["no_verbatim_gold_in_served"] += 1
                print("[e2swap] skip %s q%s: gold %r not verbatim in served rows" % (cid, it["qi"], gold[:30]),
                      flush=True)
                continue

            # ---- menu: negative + order chosen by the no-context prior (kv_e2hot2 verbatim) ----
            menu, best = None, None
            for neg in NEGS.get(qkey(it["conv"], str(it["qi"]), it["question"]), [])[: args.pool]:
                for ga in (True, False):
                    a, b = (gold, neg) if ga else (neg, gold)
                    c, pp = assemble(llm, [(H_kv, HPOS)])
                    pa, pb = probe(c, pp, llm._ids(recog_q(it["question"], a, b) + tail + "Answer:"))
                    del c
                    tot = pa + pb
                    pg = (pa if ga else pb) / tot if tot > 0 else 0.5
                    if best is None or abs(pg - .5) < abs(best[0] - .5):
                        best = (pg, neg, ga, pa, pb)
            if best is None:
                skipped["no_negs"] += 1
                continue
            pg0, neg, ga, pa0, pb0 = best
            a, b = (gold, neg) if ga else (neg, gold)
            menu = (llm._ids(recog_q(it["question"], a, b) + tail + "Answer:"), ga)
            if pat.search(neg):
                skipped["neg_contains_gold"] += 1
                continue

            # ---- counterfactual world: v -> v' in EVERY turn (R3, conv scope) ----
            cf_turns = [pat.sub(neg, t) for t in turns]
            changed = [i for i in range(NT) if cf_turns[i] != turns[i]]
            residual = sum(len(pat.findall(t)) for t in cf_turns)
            assert residual == 0, "residual gold after rewrite: %d" % residual
            eids_c, spans_c, flat_c, cur_c = world(cf_turns)
            full_kv_c = encode_block(llm, H_ids + flat_c, list(range(cur_c)), keep_a=H)

            V = VSET.get((cid, int(it["qi"])), set())
            rec = {k: it[k] for k in ("conv", "qi", "cat", "gold", "serve", "placebo",
                                      "nV_served", "nV_used")}
            rec.update(neg=neg, gold_is_a=int(ga), r_none_p=round(pg0, 4),
                       r_none_lo=round(log_odds(pa0, pb0, ga), 4),
                       n_changed=len(changed), srv_hit=srv_hit,
                       p_changed=int(any(i in set(changed) for i in pl)),
                       tok_drift=cur_c - cur)
            for h in HOTS:
                rec["v_in_hot_%d" % h] = int(bool(V & set(HOT[h])))
                rec["hot_changed_%d" % h] = int(any(i in set(changed) for i in HOT[h]))

            for _k, _h in CELLS:
                hot_u = HOT[_h]
                for _tag, _base in (("", srv[:_k]), ("p_", pl[:_k])):
                    units = sorted(set(_base) | set(hot_u))
                    if not units:
                        continue
                    for wtag, (fk, ei, sp) in (("o_", (full_kv, eids, spans)),
                                               ("c_", (full_kv_c, eids_c, spans_c))):
                        for m in ARMS:
                            p_, lo_ = rp(fk, ei, sp, units, m, hot_u, menu)
                            rec["c%d_%d_%s%s%s_p" % (_k, _h, wtag, _tag, m)] = p_
                            rec["c%d_%d_%s%s%s_lo" % (_k, _h, wtag, _tag, m)] = lo_
                        if _h == 0:
                            a_ = rec["c%d_%d_%s%sbhot_lo" % (_k, _h, wtag, _tag)]
                            b_ = rec["c%d_%d_%s%sharv_lo" % (_k, _h, wtag, _tag)]
                            assert a_ == b_, "bhot != harv at hot=0 (%s q%s %s%s)" % (cid, it["qi"], wtag, _tag)
                torch.cuda.empty_cache()
            del full_kv_c
            torch.cuda.empty_cache()
            rows.append(rec)
            print("[e2swap] %s q%s | " % (cid, it["qi"])
                  + " ".join("k%dh%d %s" % (_k, _h, " ".join(
                      "%s=%.4f" % (w + t + m, rec.get("c%d_%d_%s%s%s_lo" % (_k, _h, w, t, m), float("nan")))
                      for w in ("o_", "c_") for t in ("", "p_") for m in ARMS))
                      for _k, _h in CELLS)
                  + " | vhot=%s hchg=%s chg=%d pchg=%d drift=%d gold=%s neg=%s"
                  % ("".join(str(rec["v_in_hot_%d" % h]) for h in HOTS),
                     "".join(str(rec["hot_changed_%d" % h]) for h in HOTS),
                     len(changed), rec["p_changed"], rec["tok_drift"], gold[:20], neg[:20]),
                  flush=True)
            if args.limit and len(rows) >= args.limit:
                break
        del full_kv
        torch.cuda.empty_cache()
        if args.limit and len(rows) >= args.limit:
            break

    json.dump({"sets": args.sets, "cells": CELLS, "hots": HOTS, "rows": rows,
               "skipped": dict(skipped)}, open(args.out, "w"), indent=1)
    n = len(rows)
    print("[e2swap] skipped: %s" % dict(skipped))
    if n:
        import statistics as st
        print("\n===== E2-SWAP (n=%d, %d conversations; this shard only) =====" % (n, len({r['conv'] for r in rows})))
        for _k, _h in CELLS:
            use = [r for r in rows if not r.get("v_in_hot_%d" % _h)]
            def m(key):
                v = [r[key] for r in use if key in r]
                return st.fmean(v) if v else float("nan")
            ob, os_, cb, cs = ("c%d_%d_o_bhot_lo" % (_k, _h), "c%d_%d_o_self_lo" % (_k, _h),
                               "c%d_%d_c_bhot_lo" % (_k, _h), "c%d_%d_c_self_lo" % (_k, _h))
            d_orig = st.fmean([r[ob] - r[os_] for r in use]) if use else float("nan")
            d_cf = st.fmean([r[cb] - r[cs] for r in use]) if use else float("nan")
            neg_rate = st.fmean([1.0 if r[cs] < 0 else 0.0 for r in use]) if use else float("nan")
            print("  k=%d hot=%-2d n=%3d  (bhot-self)_orig %+7.3f  (bhot-self)_cf %+7.3f  I %+7.3f"
                  "  | levels o_bhot %+6.2f o_self %+6.2f c_bhot %+6.2f c_self %+6.2f | cf self<0: %.2f"
                  % (_k, _h, len(use), d_orig, d_cf, d_orig - d_cf, m(ob), m(os_), m(cb), m(cs), neg_rate))
        print("  peak GPU alloc %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30))
    print("E2SWAP_DONE", flush=True)


if __name__ == "__main__":
    main()
