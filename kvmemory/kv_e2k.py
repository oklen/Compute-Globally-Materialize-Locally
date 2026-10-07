"""kv_e2k.py -- variant of kv_e2.py that adds the budget sweep k = 1/2/3/5 on the log-odds readout
(read by analysis/an1.py ... an10.py and an_e2k.py).

Parent module's description:
kv_e2.py -- E2: the composite headline. On the rows a deployed retriever picks when it is
FORCED to work without any value row, is spliced KV worth more than the same text re-encoded?

    PRIMARY   harv - iso, paired, cluster = conversation
    CONTROL   the same difference on a PLACEBO serve set (rows the same retriever picks for a
              DIFFERENT question of the same conversation).  If the placebo moves too, the benefit
              is generic contextualisation and not about this question's omitted value.

Arms (identical question and readout; carrier TEXT identical across text/iso/harv):
    none    header + question                                  floor
    text    the serve rows as visible text                     RAG reference
    iso     the serve rows' KV, each run encoded ALONE at its original positions
    harv    the serve rows' KV gathered from the ONE full-conversation prefill
    p_iso / p_harv   the same two, on the placebo serve set

Readout is kv_harvest3's, unchanged: "Answer with as few words as possible", 24-token greedy,
same-model judge, every answer saved for cross-family re-judging.  The paired difference is what
we report, so "the text contains the answer" and "guessed it from the prior" cancel.

Delta is decomposed into three cells (abstain->gold, abstain->wrong, wrong->gold).  Only the third
is unambiguous evidence of a state transfer; if the gain is all in the first, the honest reading is
that harvesting activated the readout, not that it carried the value -- that is exactly what the
Gemma-4 positive control did in X8 and it must not be reported as the same thing.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e2 --sets ./out/e2_sets.json \\
        --data ./data/locomo10.json --out ./out/e2.json
"""
from __future__ import annotations

# E2, swept over the number of served rows.
#
# Why this sweep is the decisive one.  The full-set run showed harv - iso = +1.00 log-odds
# (p=.002) and split it as
#
#     self - iso  = +1.04   the served rows encoded TOGETHER rather than one at a time
#     harv - self = -0.04   everything else in the conversation
#
# At k=1 the first channel cannot exist -- one row has nothing to be encoded together with, so
# `self` and `iso` are the identical computation and their difference must be exactly 0.0.  That
# makes k=1 two things at once:
#
#   * a null-by-construction check on the harness (any nonzero self-iso at k=1 is a bug), and
#   * the clean measurement of materialization in the strict sense: harv - iso at k=1 is a single
#     row encoded inside the whole conversation versus the same row encoded alone, with no
#     joint-encoding channel to confound it.
#
# The sweep then says whether the joint-encoding gain scales with the budget, which is the
# production knob.  Generation and the judge are dropped: they cost most of the runtime and the
# full-set run already showed judged accuracy cannot resolve this effect (+0.008, p=.63, on the
# same items where log-odds gives +1.00, p=.002).

import argparse
import math
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
from kvmemory.cross_judge import judge
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest4 import recog_q
from kvmemory.kv_e3negs import qkey
from kvmemory.locomo import load_locomo
from kvmemory.rA_roles import _utterance

INSTR = "Answer with as few words as possible (a name, a date, a short phrase)."
_ABSTAIN = re.compile(
    r"\b(i (don'?t|do not) know|not (mentioned|stated|specified|provided|available|given)|"
    r"no (information|mention|answer|record)|cannot (be )?(determine|answer|tell)|"
    r"can'?t (determine|answer|tell)|unknown|unclear|n/?a|none of the)\b", re.I)


def log_odds(pa, pb, gold_is_a, floor=1e-12):
    """Unbounded companion to p_gold, from the same two candidate masses.

    E2's headline arms are scored by a JUDGE -- one bit per item -- which is the coarsest
    readout in the project.  E3 showed what that costs: on identical data the paired effect
    was invisible on a bounded scale (median exactly 0.0000, sign test p=.07) and clearly
    present on log-odds (median +0.25, p=.0035).  If E2's conversation channel is a small
    per-item shift rather than an answer flip, binary accuracy cannot see it by construction.
    Same probe, same forward pass -- only what gets recorded changes."""
    g, n = (pa, pb) if gold_is_a else (pb, pa)
    return math.log(max(g, floor)) - math.log(max(n, floor))


def cell(ans, ok):
    if ok:
        return "gold"
    if not ans.strip() or _ABSTAIN.search(ans):
        return "abstain"
    return "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--negs", default="",
                    help="kv_e2negs output. Enables the forced-choice co-readout, which reads "
                         "probability mass instead of a coin flip -- the measured power of the "
                         "binary primary is only 25-36%% at delta=.02 on this cluster structure.")
    ap.add_argument("--ks", default="1,2,3,5",
                    help="serve-set sizes to sweep; prefixes of the ranked serve list")
    ap.add_argument("--pool", type=int, default=8, help="negatives probed per item")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S = json.load(open(args.sets))
    items = S["items"]
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    NEGS = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    if args.negs:
        print(f"[e2] forced-choice readout on {len(NEGS)} items", flush=True)
    HPOS = list(range(H))

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        pa, pb = cand.get("A", 0.0), cand.get("B", 0.0)
        return pa, pb

    by_conv = defaultdict(list)
    for it in items:
        by_conv[it["conv"]].append(it)
    convs = [c for i, c in enumerate(sorted(by_conv)) if i % args.nshards == args.shard]

    KS = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    print("[e2k] sweeping k = %s" % KS, flush=True)
    rows = []
    for cid in convs:
        ep = eps[cid]
        turns = [_utterance(s) for s in ep.segments]
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids) for t in turns]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        print(f"[e2] conv {cid} turns={len(turns)} tok={cur} items={len(by_conv[cid])}", flush=True)
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)

        for it in by_conv[cid]:
            qt = f"\n\nQuestion: {it['question']}\n{INSTR}" + tail
            qids = llm._ids(qt)
            rec = {k: it[k] for k in ("conv", "qi", "cat", "gold", "serve", "placebo",
                                      "nV_used", "serve_has_goldev")}

            def gen_text(body):
                from transformers import DynamicCache
                o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(head + body + qt), 24)
                return o.strip()

            def harv_of(units):
                crows = [pp for i in units for pp in range(*spans[i])]
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, qids, 24)
                del c, sub
                return o.strip()

            def iso_of(units):
                parts, runs, cur_run = [], [], [units[0]]
                for i in units[1:]:
                    if i == cur_run[-1] + 1:
                        cur_run.append(i)
                    else:
                        runs.append(cur_run)
                        cur_run = [i]
                runs.append(cur_run)
                for run in runs:
                    a, b = spans[run[0]][0], spans[run[-1]][1]
                    ids_run = [t for i in run for t in eids[i]]
                    parts.append((encode_block(llm, ids_run, list(range(a, b))),
                                  list(range(a, b))))
                c, p = assemble(llm, [(H_kv, list(range(H)))] + parts)
                o, _, _ = llm._greedy_pos(c, p, qids, 24)
                del c, parts
                return o.strip()

            def self_of(units):
                """FLOOR ARM (new in kv_e2floor).  Prefill ONLY the served rows -- all of them in
                one block, at their original positions -- then gather and splice through the same
                path harv uses.  It sits exactly between the two published arms:

                    iso        each contiguous run encoded alone
                    self       the served rows encoded together, nothing else
                    harv       the served rows encoded inside the WHOLE conversation

                So `harv - self` is the ONLY channel through which anything the omitted source
                deposited could arrive, while `self - iso` collects everything that is NOT about
                the omitted source: the rows seeing each other, and the chunked-prefill + gather
                code path.  The published PRIMARY (harv - iso) sums the two and cannot tell them
                apart -- which matters because its placebo moved with it, i.e. most of what it
                measured was content-independent."""
                crows = [pp for i in units for pp in range(*spans[i])]
                ids_sel = [t for i in units for t in eids[i]]
                kv_sel = encode_block(llm, ids_sel, crows)
                c, p = assemble(llm, [(H_kv, list(range(H))), (kv_sel, crows)])
                o, _, _ = llm._greedy_pos(c, p, qids, 24)
                del c, kv_sel
                return o.strip()

            srv = sorted(it["serve"])
            pl = sorted(it["placebo"])

            # ---- forced-choice menu, negative and order chosen by the NO-CONTEXT prior only ----
            menu = None
            cands = NEGS.get(qkey(it["conv"], str(it["qi"]), it["question"]), [])
            if cands:
                best = None
                for neg in cands[: args.pool]:
                    for ga in (True, False):
                        a, b = (it["gold"], neg) if ga else (neg, it["gold"])
                        c, pp = assemble(llm, [(H_kv, HPOS)])
                        pa, pb = probe(c, pp, llm._ids(recog_q(it["question"], a, b) + tail
                                                       + "Answer:"))
                        del c
                        tot = pa + pb
                        pg = (pa if ga else pb) / tot if tot > 0 else 0.5
                        if best is None or abs(pg - .5) < abs(best[0] - .5):
                            best = (pg, neg, ga, pa, pb)
                if best:
                    pg0, neg, ga, pa0, pb0 = best
                    a, b = (it["gold"], neg) if ga else (neg, it["gold"])
                    menu = (llm._ids(recog_q(it["question"], a, b) + tail + "Answer:"), ga)
                    rec["neg"] = neg
                    rec["gold_is_a"] = int(ga)
                    rec["r_none_p"] = round(pg0, 4)
                    rec["r_none_lo"] = round(log_odds(pa0, pb0, ga), 4)

            def rp(units, mode):
                """forced-choice readout on the same served rows.

                mode is one of harv / self / iso -- the same three constructions the generative
                arms use, so the split (self-iso = path, harv-self = the conversation channel)
                is available on the sensitive scale too, not just on judged accuracy."""
                if menu is None:
                    return None, None
                if mode == "self":
                    crows = [pp for i in units for pp in range(*spans[i])]
                    blocks = [(H_kv, HPOS),
                              (encode_block(llm, [t for i in units for t in eids[i]], crows),
                               crows)]
                elif mode == "harv":
                    crows = [pp for i in units for pp in range(*spans[i])]
                    rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                    blocks = [(H_kv, HPOS),
                              ([(K.index_select(2, rt), V.index_select(2, rt))
                                for K, V in full_kv], crows)]
                else:
                    parts, runs, cur_run = [], [], [units[0]]
                    for i in units[1:]:
                        if i == cur_run[-1] + 1:
                            cur_run.append(i)
                        else:
                            runs.append(cur_run)
                            cur_run = [i]
                    runs.append(cur_run)
                    for run in runs:
                        aa, bb = spans[run[0]][0], spans[run[-1]][1]
                        parts.append((encode_block(llm, [t for i in run for t in eids[i]],
                                                   list(range(aa, bb))), list(range(aa, bb))))
                    blocks = [(H_kv, HPOS)] + parts
                c, pp = assemble(llm, blocks)
                pa, pb = probe(c, pp, menu[0])
                del c
                tot = pa + pb
                return (round((pa if menu[1] else pb) / tot if tot > 0 else 0.5, 4),
                        round(log_odds(pa, pb, menu[1]), 4))
            if menu is None:
                continue
            for _k in KS:
                for _tag, _units in (("", srv[:_k]), ("p_", pl[:_k])):
                    if not _units:
                        continue
                    for _m in ("harv", "self", "iso"):
                        _p, _lo = rp(_units, _m)
                        rec["k%d_%s%s_p" % (_k, _tag, _m)] = _p
                        rec["k%d_%s%s_lo" % (_k, _tag, _m)] = _lo
                torch.cuda.empty_cache()
            rows.append(rec)
            print("[e2k] %s q%s | " % (cid, it["qi"])
                  + " ".join("k%d %s" % (_k, " ".join(
                      "%s=%.4f" % (m, rec.get("k%d_%s_lo" % (_k, m), float("nan")))
                      for m in ("iso", "self", "harv", "p_iso", "p_self", "p_harv")))
                      for _k in KS)
                  + " | gold=%s" % it["gold"][:20], flush=True)
            if args.limit and len(rows) >= args.limit:
                break
        del full_kv
        torch.cuda.empty_cache()
        if args.limit and len(rows) >= args.limit:
            break

    json.dump({"sets": args.sets, "arm": S["arm"], "ks": KS, "rows": rows},
              open(args.out, "w"), indent=1)
    n = len(rows)
    if n:
        import statistics as st
        print("\n===== E2-K (n=%d, %d conversations) =====" % (n, len(set(r["conv"] for r in rows))))
        for _k in KS:
            def d(hi, lo):
                v = [r["k%d_%s_lo" % (_k, hi)] - r["k%d_%s_lo" % (_k, lo)] for r in rows
                     if "k%d_%s_lo" % (_k, hi) in r and "k%d_%s_lo" % (_k, lo) in r]
                return st.fmean(v) if v else float("nan")
            print("  k=%-2d  harv-iso %+7.4f  self-iso %+7.4f  harv-self %+7.4f "
                  "| placebo  %+7.4f %+7.4f %+7.4f"
                  % (_k, d("harv", "iso"), d("self", "iso"), d("harv", "self"),
                     d("p_harv", "p_iso"), d("p_self", "p_iso"), d("p_harv", "p_self")))
        z = [abs(r["k1_self_lo"] - r["k1_iso_lo"]) for r in rows if "k1_self_lo" in r]
        if z:
            print("  NULL-BY-CONSTRUCTION at k=1: max |self-iso| = %.6f over %d items "
                  "(must be 0.0)" % (max(z), len(z)))
    print("E2K_DONE", flush=True)

if __name__ == "__main__":
    main()
