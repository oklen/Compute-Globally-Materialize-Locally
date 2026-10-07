"""kv_e2hot.py -- E2 at the DEPLOYED operating point: the same three arms, but with a hot window.

Every E2 number so far was taken at hot_turns=0 (plan hole (7): kvmemory's router serves the last
`hot_turns` turns verbatim OUTSIDE the routed budget, which would put un-retrieved rows into the
context and break the zero-value-row condition).  hot=0 is not how the system runs.  The E2 write-up
listed `hot_turns > 0` as priority-1 of the never-run conditions and gave the reason:

    with a hot window, `harv` gets the hot tail AND the served rows out of ONE coherent prefill,
    while `self` / `iso` have to re-encode and splice -- structurally where cache reuse should win.

So this module sweeps CELLS of (k, hot) instead of k alone:

    k=5 hot=0    GATE -- must reproduce e2_k.json's k5 columns BIT-FOR-BIT
    k=1 hot=0    GATE -- self and iso are the identical computation, difference must be exactly 0
    k=5 hot=4    the production default (core.py: hot_turns = 4)
    k=5 hot=16   a 4x dose.  One point cannot separate "no effect" from "not enough hot window";
                 a flat dose-response across 0 / 4 / 16 can.

The served unit set for a cell is  sorted(set(serve[:k]) | set(last `hot` turns)) -- position
preserved, so text / positions / prefix stay byte-identical across the three arms and only the
ENCODING differs, exactly as at hot=0.

V-IN-HOT EXCLUSION (the thing that would silently destroy this experiment).  The hot window is
taken verbatim, so if any value row falls inside it the answer is served and the V-excluded
protocol is void.  We rebuild the V set here from the same two sources kv_e2sets used
(rA_labels V["N1"] union rA_loose V_loose) and flag every item per hot size.  Measured cost is
small -- 4/1269 at hot=2, 15 at hot=4, 64 at hot=16 -- but "small" is not "zero" and the flag is
written into every row so the analysis filters instead of trusting the sample.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_e2hot --sets ./out/e2_sets.json \\
        --negs ./out/e2_negs.json --labels ./out/rA_labels.json \\
        --loose ./out/rA_loose.json --cells 5:0,1:0,5:4,5:16 --out ./out/e2hot.json

--- original kv_e2k.py docstring follows -------------------------------------------------------

E2: the composite headline. On the rows a deployed retriever picks when it is
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
    ap.add_argument("--cells", default="5:0,1:0,5:4,5:16",
                    help="k:hot cells to run. k=prefix of the ranked serve list, hot=how many "
                         "trailing turns are served verbatim on top of it. The two hot=0 cells "
                         "are gates, not results: they must reproduce e2_k.json bit-for-bit.")
    ap.add_argument("--labels", default="./out/rA_labels.json")
    ap.add_argument("--loose", default="./out/rA_loose.json")
    ap.add_argument("--done_from", default="",
                    help="colon-separated globs of earlier logs; items already printed there are "
                         "skipped. A reclaimed GPU then costs only the in-flight item.")
    ap.add_argument("--pool", type=int, default=8, help="negatives probed per item")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    CELLS = []
    for tok in args.cells.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, h = tok.partition(":")
        CELLS.append((int(k), int(h or 0)))
    assert CELLS, "--cells is empty"
    assert any(h == 0 for _, h in CELLS), "at least one hot=0 cell is required as the gate"
    HOTS = sorted({h for _, h in CELLS})

    # ---- V sets, rebuilt from the SAME two sources kv_e2sets used ----
    VSET = {}
    for r in json.load(open(args.labels))["records"]:
        VSET[(str(r["conv"]), int(r["qi"]))] = set(r["V"]["N1"])
    nloose = 0
    for r in json.load(open(args.loose))["records"]:
        key = (str(r["conv"]), int(r["qi"]))
        if key in VSET:
            VSET[key] |= set(r.get("V_loose") or [])
            nloose += 1
    print("[e2hot] V sets: %d questions (%d also carry the loose paraphrase channel)"
          % (len(VSET), nloose), flush=True)

    DONE = set()
    if args.done_from:
        import glob as _glob
        import re as _re
        pat = _re.compile(r"^\[e2hot\] (\S+) q(\d+) \|")
        nf = 0
        for g in args.done_from.split(":"):
            for f in _glob.glob(g):
                nf += 1
                for line in open(f, errors="ignore"):
                    m = pat.match(line)
                    if m:
                        DONE.add((m.group(1), int(m.group(2))))
        print("[e2hot] resume: %d log files, %d items already measured" % (nf, len(DONE)),
              flush=True)

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

    print("[e2hot] cells (k, hot) = %s" % CELLS, flush=True)
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
        NT = len(turns)
        # kv_e2sets wrote n_turns per item; if the loader disagrees the hot window would be taken
        # from the wrong end of the conversation and NOTHING downstream would notice.
        for _it in by_conv[cid]:
            assert _it["n_turns"] == NT, (
                "n_turns mismatch conv %s: sets=%d loader=%d" % (cid, _it["n_turns"], NT))
        HOT = {h: list(range(max(0, NT - h), NT)) for h in HOTS}
        print(f"[e2hot] conv {cid} turns={NT} tok={cur} items={len(by_conv[cid])} "
              f"hot={ {h: (v[0], v[-1]) for h, v in HOT.items() if v} }", flush=True)
        full_kv = encode_block(llm, H_ids + flat, list(range(cur)), keep_a=H)

        for it in by_conv[cid]:
            if (cid, int(it["qi"])) in DONE:
                continue
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
            V = VSET.get((cid, int(it["qi"])))
            assert V is not None, "no V set for %s q%s" % (cid, it["qi"])
            for h in HOTS:
                rec["v_in_hot_%d" % h] = int(bool(V & set(HOT[h])))
            rec["n_turns_seen"] = NT

            for _k, _h in CELLS:
                hot_u = HOT[_h]
                for _tag, _base in (("", srv[:_k]), ("p_", pl[:_k])):
                    # union, position preserved.  serve and hot DO overlap on some items
                    # (38/1269 at hot=4); serving a row twice would change the token stream and
                    # break the byte-identity the three arms rest on.
                    _units = sorted(set(_base) | set(hot_u))
                    if not _units:
                        continue
                    for _m in ("harv", "self", "iso"):
                        _p, _lo = rp(_units, _m)
                        rec["c%d_%d_%s%s_p" % (_k, _h, _tag, _m)] = _p
                        rec["c%d_%d_%s%s_lo" % (_k, _h, _tag, _m)] = _lo
                    rec["c%d_%d_%snunit" % (_k, _h, _tag)] = len(_units)
                torch.cuda.empty_cache()
            rows.append(rec)
            print("[e2hot] %s q%s | " % (cid, it["qi"])
                  + " ".join("k%dh%d %s" % (_k, _h, " ".join(
                      "%s=%.4f" % (m, rec.get("c%d_%d_%s_lo" % (_k, _h, m), float("nan")))
                      for m in ("iso", "self", "harv", "p_iso", "p_self", "p_harv")))
                      for _k, _h in CELLS)
                  + " | vhot=%s nu=%d gold=%s"
                  % ("".join(str(rec["v_in_hot_%d" % h]) for h in HOTS),
                     rec.get("c%d_%d_nunit" % CELLS[0], -1), it["gold"][:20]), flush=True)
            if args.limit and len(rows) >= args.limit:
                break
        del full_kv
        torch.cuda.empty_cache()
        if args.limit and len(rows) >= args.limit:
            break

    json.dump({"sets": args.sets, "arm": S["arm"], "cells": CELLS, "hots": HOTS, "rows": rows},
              open(args.out, "w"), indent=1)
    n = len(rows)
    if n:
        import statistics as st
        print("\n===== E2-HOT (n=%d, %d conversations) ====="
              % (n, len(set(r["conv"] for r in rows))))
        print("  (this session only; the real read-out joins all logs and drops v_in_hot items)")
        for _k, _h in CELLS:
            use = [r for r in rows if not r.get("v_in_hot_%d" % _h)]

            def d(hi, lo):
                a, b = "c%d_%d_%s_lo" % (_k, _h, hi), "c%d_%d_%s_lo" % (_k, _h, lo)
                v = [r[a] - r[b] for r in use if a in r and b in r]
                return st.fmean(v) if v else float("nan")
            print("  k=%d hot=%-2d n=%3d  harv-iso %+7.4f  self-iso %+7.4f  harv-self %+7.4f "
                  "| placebo  %+7.4f %+7.4f %+7.4f"
                  % (_k, _h, len(use), d("harv", "iso"), d("self", "iso"), d("harv", "self"),
                     d("p_harv", "p_iso"), d("p_self", "p_iso"), d("p_harv", "p_self")))
        for _k, _h in CELLS:
            if _k != 1 or _h != 0:
                continue
            z = [abs(r["c1_0_self_lo"] - r["c1_0_iso_lo"]) for r in rows if "c1_0_self_lo" in r]
            if z:
                print("  NULL-BY-CONSTRUCTION at k=1 hot=0: max |self-iso| = %.6f over %d items "
                      "(must be 0.0)" % (max(z), len(z)))
        print("  peak GPU alloc %.1f GiB / reserved %.1f GiB"
              % (torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.max_memory_reserved() / 2**30))
    print("E2HOT_DONE", flush=True)

if __name__ == "__main__":
    main()
