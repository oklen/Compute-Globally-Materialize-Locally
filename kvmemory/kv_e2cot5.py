"""kv_e2cot5.py -- variant of kv_e2.py that adds the chain-of-thought readout
(read by analysis/cotparse2.py, cotres2.py, cotrob3.py, chunkaudit.py).

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
    ap.add_argument("--done_from", default="",
                    help="colon-separated logs of earlier runs; every item whose [e2cot] line "
                         "is already there is skipped.  A reclaimed GPU then costs only the "
                         "item in flight, not the shard.")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth item of the (already sorted) list.  Deterministic and "
                         "content-blind: a scale knob, not a filter on the outcome.")
    ap.add_argument("--cot_max", type=int, default=512,
                    help="CoT decode budget.  Truncation silently ruins this kind of run, "
                         "so the cap-hit rate is recorded per item and gated in the smoke.")
    ap.add_argument("--cot_ks", default="5",
                    help="which k values get the (expensive) CoT readout")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S = json.load(open(args.sets))
    items = S["items"]
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    _S1, _S2, _S3 = "<<<U1>>>", "<<<A1>>>", "<<<U2>>>"
    try:
        _full2 = llm.tok.apply_chat_template(
            [{"role": "user", "content": _S1}, {"role": "assistant", "content": _S2},
             {"role": "user", "content": _S3}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        _full2 = llm.tok.apply_chat_template(
            [{"role": "user", "content": _S1}, {"role": "assistant", "content": _S2},
             {"role": "user", "content": _S3}],
            tokenize=False, add_generation_prompt=True)
    _pre, _r = _full2.split(_S1)
    MID1, _r = _r.split(_S2)
    MID2, TAIL2 = _r.split(_S3)
    assert _pre == head, "SCAFFOLD_HEAD_MISMATCH -- H_kv was built from a different prefix"
    assert TAIL2 == tail, "SCAFFOLD_TAIL_MISMATCH -- the generation prompt differs between templates"
    # MID1 is the opener of a COMPLETED assistant message (no think block); the GENERATION
    # prompt is `tail` (with the empty think block).  Generating after MID1 would put the model
    # in the slot for text that is already written -- use `tail`.
    print("[e2cot2] scaffold ok: MID2=%r TAIL2=%r" % (MID2[:60], TAIL2[:60]), flush=True)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    eps = {str(e.conv_id): e for e in load_locomo(args.data)}
    NEGS = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    if args.negs:
        print(f"[e2] forced-choice readout on {len(NEGS)} items", flush=True)
    HPOS = list(range(H))

    BRIDGE_IDS = llm._ids(
        MID2 + "Based on your reasoning, which option is correct? You must reply with exactly "
               "one letter and nothing else: A or B. Even if you are unsure, pick the more "
               "likely one." + TAIL2 + "Answer:")

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        pa, pb = cand.get("A", 0.0), cand.get("B", 0.0)
        return pa, pb

    by_conv = defaultdict(list)
    for it in items:
        by_conv[it["conv"]].append(it)
    convs = [c for i, c in enumerate(sorted(by_conv)) if i % args.nshards == args.shard]
    if args.done_from:
        import glob as _glob, re as _re
        _pat = _re.compile(r"^\[e2cot\] (\S+) q(\d+)")
        DONE = set()
        for _g in args.done_from.split(":"):
            for _p in _glob.glob(_g):
                for _ln in open(_p, errors="ignore"):
                    _m = _pat.match(_ln)
                    if _m:
                        DONE.add((_m.group(1), int(_m.group(2))))
        _before = sum(len(v) for v in by_conv.values())
        for _c in list(by_conv):
            by_conv[_c] = [x for x in by_conv[_c] if (str(_c), x["qi"]) not in DONE]
        print("[e2cot5] resume: %d items already measured, %d -> %d remaining"
              % (len(DONE), _before, sum(len(v) for v in by_conv.values())), flush=True)
    if args.stride > 1:
        for _c in by_conv:
            by_conv[_c] = [x for j, x in enumerate(by_conv[_c]) if j % args.stride == 0]
        print("[e2cot4] stride=%d -> %d items kept"
              % (args.stride, sum(len(v) for v in by_conv.values())), flush=True)

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
            menu_cot = None
            ANS_IDS = None
            ga = True
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
                    menu_cot = llm._ids(
                        recog_q(it["question"], a, b)
                        + "\nThink step by step about what the conversation above says. "
                          "Do not give the final letter yet." + tail)
                    ANS_IDS = BRIDGE_IDS
                    rec["neg"] = neg
                    rec["gold_is_a"] = int(ga)
                    rec["r_none_p"] = round(pg0, 4)
                    rec["r_none_lo"] = round(log_odds(pa0, pb0, ga), 4)

            def mk_blocks(units, mode):
                """exactly what rp() used to build inline; shared so the CoT readout cannot
                drift from the probe readout."""
                if mode == "self":
                    crows = [pp for i in units for pp in range(*spans[i])]
                    return [(H_kv, HPOS),
                            (encode_block(llm, [t for i in units for t in eids[i]], crows), crows)]
                if mode == "harv":
                    crows = [pp for i in units for pp in range(*spans[i])]
                    rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                    return [(H_kv, HPOS),
                            ([(K.index_select(2, rt), V.index_select(2, rt))
                              for K, V in full_kv], crows)]
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
                return [(H_kv, HPOS)] + parts

            def cot_readout(blocks, q_ids, ans_ids, max_new):
                """Generate a chain of thought against the assembled cache, THEN read the A/B
                distribution.  One pass: the model call appends to the cache in place (that is
                why the probe path has to del its cache), so the answer tokens are fed against
                the cache the CoT just extended -- no rebuild, no drift."""
                c, pos = assemble(llm, blocks)
                dev = llm.device
                cl = pos.shape[0]
                L = q_ids.shape[1]
                np0 = int(pos.max().item()) + 1
                out = llm.model(input_ids=q_ids, past_key_values=c, use_cache=True,
                                position_ids=torch.arange(np0, np0 + L, device=dev).unsqueeze(0),
                                cache_position=torch.arange(cl, cl + L, device=dev),
                                attention_mask=torch.ones(1, cl + L, dtype=torch.long, device=dev))
                nxt = int(out.logits[0, -1].argmax())
                cur, cur_pos, toks, capped = cl + L, np0 + L, [], 1
                for _ in range(max_new):
                    if nxt in llm.eos_ids:
                        capped = 0
                        break
                    toks.append(nxt)
                    out = llm.model(input_ids=torch.tensor([[nxt]], device=dev),
                                    past_key_values=c, use_cache=True,
                                    position_ids=torch.tensor([[cur_pos]], device=dev),
                                    cache_position=torch.tensor([cur], device=dev),
                                    attention_mask=torch.ones(1, cur + 1, dtype=torch.long,
                                                              device=dev))
                    nxt = int(out.logits[0, -1].argmax())
                    cur += 1
                    cur_pos += 1
                La = ans_ids.shape[1]
                out = llm.model(input_ids=ans_ids, past_key_values=c, use_cache=True,
                                position_ids=torch.arange(cur_pos, cur_pos + La,
                                                          device=dev).unsqueeze(0),
                                cache_position=torch.arange(cur, cur + La, device=dev),
                                attention_mask=torch.ones(1, cur + La, dtype=torch.long,
                                                          device=dev))
                probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
                _tv, _ti = torch.topk(probs, 5)
                top = [(llm.tok.decode([int(i)]), round(float(v), 4))
                       for v, i in zip(_tv.tolist(), _ti.tolist())]
                cand = {}
                for w in ("A", "B"):
                    tot, seen = 0.0, set()
                    for v in (" " + w, w):
                        t0 = llm.tok(v, add_special_tokens=False).input_ids[0]
                        if t0 not in seen:
                            seen.add(t0)
                            tot += float(probs[t0])
                    cand[w] = tot
                del c
                # give the blocks back: without this the allocator holds ~35 GiB of churn from
                # the decode loop and only one process fits on the card.
                torch.cuda.empty_cache()
                return (cand["A"], cand["B"], len(toks), capped,
                        llm.tok.decode(toks, skip_special_tokens=True).strip(), top)

            def rp(units, mode):
                """forced-choice readout on the same served rows.

                mode is one of harv / self / iso -- the same three constructions the generative
                arms use, so the split (self-iso = path, harv-self = the conversation channel)
                is available on the sensitive scale too, not just on judged accuracy."""
                if menu is None:
                    return None, None
                blocks = mk_blocks(units, mode)
                c, pp = assemble(llm, blocks)
                pa, pb = probe(c, pp, menu[0])
                del c
                tot = pa + pb
                return (round((pa if menu[1] else pb) / tot if tot > 0 else 0.5, 4),
                        round(log_odds(pa, pb, menu[1]), 4))
            if menu is None:
                continue
            COT_KS = sorted({int(x) for x in args.cot_ks.split(",") if x.strip()})
            for _k in KS:
                for _tag, _units in (("", srv[:_k]), ("p_", pl[:_k])):
                    if not _units:
                        continue
                    for _m in ("harv", "self", "iso"):
                        _p, _lo = rp(_units, _m)
                        rec["k%d_%s%s_p" % (_k, _tag, _m)] = _p
                        rec["k%d_%s%s_lo" % (_k, _tag, _m)] = _lo
                        if _k in COT_KS and menu_cot is not None:
                            _pa, _pb, _n, _cap, _txt, _top = cot_readout(
                                mk_blocks(_units, _m), menu_cot, BRIDGE_IDS, args.cot_max)
                            _tot = _pa + _pb
                            _pref = "c%d_%s%s" % (_k, _tag, _m)
                            rec[_pref + "_p"] = round((_pa if ga else _pb) / _tot
                                                      if _tot > 0 else 0.5, 4)
                            rec[_pref + "_lo"] = round(log_odds(_pa, _pb, ga), 4)
                            rec[_pref + "_mass"] = round(_tot, 6)
                            rec[_pref + "_ntok"] = _n
                            rec[_pref + "_cap"] = _cap
                            rec[_pref + "_top"] = top_str = " ".join(
                                "%r:%.3f" % (t, p) for t, p in _top)
                            if _tag == "" and _m == "harv":
                                rec["cot_text"] = _txt[:400]
                                rec["cot_top"] = top_str
                torch.cuda.empty_cache()
            if menu_cot is not None:
                _pa, _pb, _n, _cap, _, _ntop = cot_readout([(H_kv, HPOS)], menu_cot,
                                                           BRIDGE_IDS, args.cot_max)
                rec["c_none_top"] = " ".join("%r:%.3f" % (t, p) for t, p in _ntop)
                _tot = _pa + _pb
                rec["c_none_p"] = round((_pa if ga else _pb) / _tot if _tot > 0 else 0.5, 4)
                rec["c_none_lo"] = round(log_odds(_pa, _pb, ga), 4)
                rec["c_none_mass"] = round(_tot, 6)
            rows.append(rec)
            print("[e2k] %s q%s | " % (cid, it["qi"])
                  + " ".join("k%d %s" % (_k, " ".join(
                      "%s=%.4f" % (m, rec.get("k%d_%s_lo" % (_k, m), float("nan")))
                      for m in ("iso", "self", "harv", "p_iso", "p_self", "p_harv")))
                      for _k in KS)
                  + " | gold=%s" % it["gold"][:20], flush=True)
            if menu_cot is not None:
                print("[e2cot] %s q%s | " % (cid, it["qi"])
                      + " ".join("%s=%.4f" % (m, rec.get("c5_%s_lo" % m, float("nan")))
                                 for m in ("iso", "self", "harv", "p_iso", "p_self", "p_harv"))
                      + " | none=%.4f mass=%.4f" % (rec.get("c_none_lo", float("nan")),
                                                    rec.get("c_none_mass", float("nan")))
                      + " | mass=%.4f ntok=%d cap=%d"
                      % (rec.get("c5_harv_mass", float("nan")),
                         rec.get("c5_harv_ntok", -1), rec.get("c5_harv_cap", -1))
                      + " | TOP=%s" % rec.get("cot_top", "")[:120]
                      + " | COT=%s" % rec.get("cot_text", "")[:80].replace("\n", " "),
                      flush=True)
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
    if n:
        import statistics as st
        cot = [r for r in rows if "c5_harv_lo" in r]
        if cot:
            def cd(hi, lo):
                return st.fmean([r["c5_%s_lo" % hi] - r["c5_%s_lo" % lo] for r in cot])
            def nd(hi, lo):
                return st.fmean([r["k5_%s_lo" % hi] - r["k5_%s_lo" % lo] for r in cot])
            print("\n===== CoT vs no-CoT at k=5 (n=%d) =====" % len(cot))
            print("  %-12s %10s %10s" % ("contrast", "no-CoT", "CoT"))
            for hi, lo in (("harv", "iso"), ("self", "iso"), ("harv", "self"),
                           ("p_harv", "p_iso"), ("p_self", "p_iso"), ("p_harv", "p_self")):
                print("  %-12s %+10.4f %+10.4f" % ("%s-%s" % (hi, lo), nd(hi, lo), cd(hi, lo)))
            print("  readout health: candidate mass mean %.4f min %.4f | CoT tokens mean %.1f "
                  "| hit the cap %.1f%% | none p_gold %.4f"
                  % (st.fmean([r["c5_harv_mass"] for r in cot]),
                     min(r["c5_harv_mass"] for r in cot),
                     st.fmean([r["c5_harv_ntok"] for r in cot]),
                     100.0 * st.fmean([r["c5_harv_cap"] for r in cot]),
                     st.fmean([r["c_none_p"] for r in cot if "c_none_p" in r])))
    print("E2K_DONE", flush=True)

if __name__ == "__main__":
    main()
