"""kv_region.py -- what does each REGION's KV carry?  (paper section 9, read-out)

Input: trajectories from kv_traj.py (one token stream + region spans per item).  For each item
two follow-ups are asked AFTER the trajectory, as a new user turn on top of the assistant transcript:

    stated     q1 verbatim, gold v          (v is stated in tool_out; usually copied into answer)
    unstated   "which year is mentioned in the passage about <title>?" for a RETRIEVED passage
               whose year appears in no generated region (plan / scratchpad / answer)

Q1  per region r:   full, drop_r (rows of r evicted), tx_r (rows of r re-encoded alone at their
                    positions, everything else in situ)    -> full - tx_r, full - drop_r
Q2  source region s = the tool_out where v first appears.  Counterfactual stream: v -> v' inside
    s only (R3 within s), downstream text byte-identical, positions shifted by the token drift.
    For keep-sets K (all-but-s / own-outputs / plan-only): lo(orig,K) - lo(cf,K) = what K carries
    of s.  Placebo: a different value in s rewritten.  Positive control: keep {s', system, user}.
Q3  token share per region (from the trajectories).

Readout: forced choice gold vs negative (no-context prior picks the negative), log-odds.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        python -m kvmemory.kv_region --traj ./out/traj.json --negs ./out/hnegs.json --out ./out/region.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_fail2 import answer_with_probs
from kvmemory.kv_harvest4 import recog_q
from kvmemory.kv_e3ctrl import shape_class
from kvmemory.kv_cite import load_items, wb, strip_paren
from kvmemory.kv_ans import log_odds, YEAR

REGIONS = ("system", "user", "plan", "tool_in", "tool_out", "scratchpad", "answer")
GEN = ("plan", "scratchpad", "answer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--done_from", default="")
    ap.add_argument("--negs", default="./out/hnegs.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    HN = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    print("[region] LLM negatives for %d items" % len(HN), flush=True)
    T = json.load(open(args.traj))
    trajs = [t for t in T["trajs"] if t.get("answer") and t.get("src")]
    print("[region] %d trajectories with an answer and the gold retrieved" % len(trajs), flush=True)
    if args.limit:
        trajs = trajs[: args.limit]
    items = {it["id"]: it for it in load_items(args.data)}
    by_shape = {}
    for it in items.values():
        by_shape.setdefault(shape_class(it["v"]), []).append(it["v"])
    DONE = set()
    if args.done_from:
        import glob
        for g in args.done_from.split(":"):
            for f in glob.glob(g):
                for line in open(f, errors="ignore"):
                    m = re.match(r"^\[region\] (\S+) \|", line)
                    if m:
                        DONE.add(m.group(1))

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    ids = lambda s: list(llm.tok(s, add_special_tokens=False).input_ids)

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        return cand.get("A", 0.0), cand.get("B", 0.0)

    def menu_for(q, gold, cands):
        best = None
        for neg in cands[: args.pool]:
            for ga in (True, False):
                a, b = (gold, neg) if ga else (neg, gold)
                kv = encode_block(llm, ids(head), list(range(len(ids(head)))))
                c, pp = assemble(llm, [(kv, list(range(len(ids(head)))))])
                pa, pb = probe(c, pp, llm._ids(recog_q(q, a, b) + tail + "Answer:"))
                del c
                tot = pa + pb
                pg = (pa if ga else pb) / tot if tot > 0 else 0.5
                if best is None or abs(pg - .5) < abs(best[0] - .5):
                    best = (pg, neg, ga, log_odds(pa, pb, ga))
        if best is None:
            return None
        pg, neg, ga, lo_none = best
        a, b = (gold, neg) if ga else (neg, gold)
        return dict(neg=neg, ga=ga, lo_none=round(lo_none, 4),
                    q2ids=torch.tensor([ids("<|im_end|>\n" + head + recog_q(q, a, b) + tail + "Answer:")],
                                       dtype=torch.long, device=llm.device))

    def read(cache, pos, m):
        pa, pb = probe(cache, pos, m["q2ids"])
        return round(log_odds(pa, pb, m["ga"]), 4)

    def serve(full_kv, stream, regs, keep, fresh=()):
        """assemble rows of regions in `keep` (harvested) and of regions in `fresh` (encoded alone)."""
        blocks = []
        for r in regs:
            if r["name"] in fresh:
                pos = list(range(r["s"], r["e"]))
                blocks.append((encode_block(llm, stream[r["s"]:r["e"]], pos), pos))
            elif r["name"] in keep:
                pos = list(range(r["s"], r["e"]))
                rt = torch.tensor(pos, dtype=torch.long)
                blocks.append(([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv], pos))
        blocks.sort(key=lambda b: b[1][0])
        return assemble(llm, blocks)

    rows, skip = [], Counter()
    for t in trajs:
        if t["id"] in DONE:
            continue
        it = items.get(t["id"])
        if it is None:
            skip["no_item"] += 1; continue
        q1, v = t["q"], t["v"]
        stream, regs = t["stream"], t["regions"]
        N = len(stream)
        gen_text = " ".join(r["text"] for r in regs if r["name"] in GEN)
        # ---- follow-ups ----
        alltxt = " ".join(r["text"] for r in regs)
        cands_v = [x for x in by_shape.get(shape_class(v), []) if x.lower() != v.lower()
                   and not wb(x).search(alltxt) and not wb(v).search(x)]
        rnd.shuffle(cands_v)
        llm_negs = [x for x in HN.get(t["id"], {}).get("ans", []) if x.lower() != v.lower()
                    and not wb(x).search(alltxt) and not wb(v).search(x)]
        cands_v = llm_negs + cands_v
        qs = {}
        m1 = menu_for(q1, v, list(dict.fromkeys(cands_v)))
        if m1 is not None:
            qs["stated"] = (q1, v, m1, int(bool(wb(v).search(gen_text))))
        # unstated: a year in a retrieved passage's text, absent from every generated region
        for r in [r for r in regs if r["name"] == "tool_out"]:
            for mm in re.finditer(r"\[([^\]]+)\] ", r["text"]):
                title = mm.group(1)
                seg = r["text"][mm.end(): mm.end() + 400]
                ys = YEAR.findall(seg.split(". ")[0])
                if len(ys) != 1:
                    continue
                f = ys[0]
                if wb(f).search(gen_text) or wb(f).search(q1) or f == v:
                    continue
                q2 = "In the search results shown earlier, which year is mentioned in the passage about %s?" % strip_paren(title)
                negs = [str(int(f) + d) for d in (-7, -3, 3, 7, -12, 12, -20, 20)]
                negs = [y for y in negs if not wb(y).search(" ".join(rr["text"] for rr in regs))]
                m2 = menu_for(q2, f, negs)
                if m2 is not None:
                    qs["unstated"] = (q2, f, m2, 0)
                break
            if "unstated" in qs:
                break
        if not qs:
            skip["no_followup"] += 1; continue

        full_kv = encode_block(llm, stream, list(range(N)))
        rec = dict(id=t["id"], q1=q1, v=v, n_tok=N, steps=t["steps"], answer=t["answer"],
                   answer_correct=t["answer_correct"], note_states_gold=t["note_states_gold"],
                   share={n: sum(r["e"] - r["s"] for r in regs if r["name"] == n) for n in REGIONS})
        present = {r["name"] for r in regs}
        for kind, (q2, gold, m, stated) in qs.items():
            rec[kind + "_gold"], rec[kind + "_neg"], rec[kind + "_none"], rec[kind + "_stated"] = gold, m["neg"], m["lo_none"], stated
            c, pp = serve(full_kv, stream, regs, set(REGIONS)); rec[kind + "_full"] = read(c, pp, m); del c
            for rname in REGIONS:
                if rname not in present or rname in ("system", "user"):
                    continue
                c, pp = serve(full_kv, stream, regs, set(REGIONS) - {rname}); rec["%s_drop_%s" % (kind, rname)] = read(c, pp, m); del c
                c, pp = serve(full_kv, stream, regs, set(REGIONS) - {rname}, fresh={rname}); rec["%s_tx_%s" % (kind, rname)] = read(c, pp, m); del c
            torch.cuda.empty_cache()

        # ---- Q2: counterfactual inside the source tool_out ----
        src_name, s0, e0 = t["src"]
        src_i = next(i for i, r in enumerate(regs) if r["s"] == s0 and r["e"] == e0)
        if "stated" in qs:
            q2, gold, m, _ = qs["stated"]
            pat = wb(v)
            cf_text = pat.sub(m["neg"], regs[src_i]["text"])
            assert not pat.search(cf_text)
            cf_ids = ids(cf_text + "\n")
            drift = len(cf_ids) - (e0 - s0)
            stream_cf = stream[:s0] + cf_ids + stream[e0:]
            regs_cf = []
            for i, r in enumerate(regs):
                if i < src_i:
                    regs_cf.append(dict(r))
                elif i == src_i:
                    regs_cf.append(dict(r, e=s0 + len(cf_ids), text=cf_text))
                else:
                    regs_cf.append(dict(r, s=r["s"] + drift, e=r["e"] + drift))
            full_cf = encode_block(llm, stream_cf, list(range(len(stream_cf))))
            # the other tool_outs may ALSO state v (the gold passage retrieved twice) -> flag
            rec["v_in_other_tool_out"] = int(any(pat.search(r["text"]) for i, r in enumerate(regs) if r["name"] == "tool_out" and i != src_i))
            rec["cf_drift"] = drift
            KEEPS = {"allbutsrc": set(REGIONS), "own": {"system", "user", "scratchpad", "answer"},
                     "plan": {"system", "user", "plan"}, "srconly": {"system", "user"}}
            def serve_keep(kv, st, rg, keep, with_src):
                blocks = []
                for i, r in enumerate(rg):
                    if r["name"] not in keep and not (with_src and i == src_i):
                        continue
                    if not with_src and i == src_i:
                        continue
                    pos = list(range(r["s"], r["e"])); rt = torch.tensor(pos, dtype=torch.long)
                    blocks.append(([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in kv], pos))
                blocks.sort(key=lambda b: b[1][0])
                return assemble(llm, blocks)
            for kn, keep in KEEPS.items():
                with_src = (kn == "srconly")
                c, pp = serve_keep(full_kv, stream, regs, keep, with_src); lo_o = read(c, pp, m); del c
                c, pp = serve_keep(full_cf, stream_cf, regs_cf, keep, with_src); lo_c = read(c, pp, m); del c
                rec["q2_%s_orig" % kn], rec["q2_%s_cf" % kn], rec["q2_%s_d" % kn] = lo_o, lo_c, round(lo_o - lo_c, 4)
            # placebo: rewrite a year (not v) inside the source passage
            ys = [y for y in YEAR.findall(regs[src_i]["text"]) if y != v and not wb(y).search(gen_text)]
            if ys:
                y = ys[0]; y2 = str(int(y) + 3)
                pl_text = wb(y).sub(y2, regs[src_i]["text"])
                pl_ids = ids(pl_text + "\n"); d2 = len(pl_ids) - (e0 - s0)
                stream_pl = stream[:s0] + pl_ids + stream[e0:]
                regs_pl = [dict(r) if i < src_i else (dict(r, e=s0 + len(pl_ids), text=pl_text) if i == src_i else dict(r, s=r["s"] + d2, e=r["e"] + d2)) for i, r in enumerate(regs)]
                full_pl = encode_block(llm, stream_pl, list(range(len(stream_pl))))
                c, pp = serve_keep(full_pl, stream_pl, regs_pl, KEEPS["allbutsrc"], False); lo_p = read(c, pp, m); del c
                rec["q2_placebo_d"] = round(rec["q2_allbutsrc_orig"] - lo_p, 4)
                del full_pl
            del full_cf
        del full_kv
        torch.cuda.empty_cache()
        rows.append(rec)
        print("[region] %s | " % t["id"] + " ".join(
            "%s: none=%+.2f full=%+.2f " % (k, rec[k + "_none"], rec[k + "_full"]) + " ".join(
                "%s:d%+.2f/t%+.2f" % (rn[:4], rec[k + "_full"] - rec["%s_drop_%s" % (k, rn)], rec[k + "_full"] - rec["%s_tx_%s" % (k, rn)])
                for rn in REGIONS if "%s_drop_%s" % (k, rn) in rec) for k in qs)
            + " | q2 allbutsrc %s own %s plan %s srconly %s placebo %s"
            % tuple(("%+.2f" % rec[x]) if x in rec else "na" for x in ("q2_allbutsrc_d", "q2_own_d", "q2_plan_d", "q2_srconly_d", "q2_placebo_d")), flush=True)

    json.dump({"traj": args.traj, "rows": rows, "skip": dict(skip)}, open(args.out, "w"), indent=1)
    print("[region] rows=%d skip=%s" % (len(rows), dict(skip)))
    print("REGION_DONE", flush=True)


if __name__ == "__main__":
    main()
