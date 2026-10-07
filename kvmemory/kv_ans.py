"""kv_ans.py -- the previous answer's KV as session memory (paper section 9).

Turn 1: q1 + 10 passages D1 -> the model writes a1.  Turn 2: a follow-up q2.  Can we EVICT D1 and
keep only a1's KV (harvested from the turn-1 prefill, where a1 was computed attending to D1)?

Arms for q2 (a1's tokens and positions identical across full / ansKV / ansTX):
    full      keep the whole turn-1 KV (header, D1, q1, a1)                     ceiling
    ansKV     keep header + q1 + a1 rows HARVESTED from the turn-1 prefill; drop D1   (= bhot/harv)
    ansTX     the same rows RE-ENCODED jointly at their original positions, no D1    (= self)
    thinkKV   as ansKV but the thinking tokens are kept too (thinking-mode generation, subset)
    rerank    text: top-3 BM25 passages of D1 for q2                                 production
    none      q2 alone
    p_ansKV   another item's a1 rows (harvested from THAT item's prefill)            placebo/floor

Follow-ups:
    (i)  stated    q2 = q1 verbatim ("remind me")            gold = v      a1 states it
    (ii) unstated  q2 asks the year mentioned in a distractor passage d_F  gold = that year;
                   required NOT to appear in a1

Readout: forced choice gold vs negative, log-odds, negative chosen by the no-context prior.
The contract's prediction: (i) ansKV > ansTX (narrow form); (ii) ansKV ~ ansTX ~ none, rerank wins;
thinkKV is the only arm that could move (ii).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        python -m kvmemory.kv_ans --negs ./out/hnegs.json --out ./out/ans.json
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

INSTR1 = "Answer the question in one or two sentences using the passages."
INSTR2 = "Answer with as few words as possible (a name, a date, a short phrase)."
YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")


def log_odds(pa, pb, gold_is_a, floor=1e-12):
    g, n = (pa, pb) if gold_is_a else (pb, pa)
    return math.log(max(g, floor)) - math.log(max(n, floor))


def bm25_top(passages, q, k=3):
    toks = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    qt = set(toks(q))
    N = len(passages)
    df = Counter()
    docs = [toks(p) for _, p in passages]
    for d in docs:
        for w in set(d):
            df[w] += 1
    avg = sum(len(d) for d in docs) / max(1, N)
    sc = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        sc.append((s, i))
    return [passages[i] for _, i in sorted(sc, reverse=True)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hotpotqa/hotpot_dev_distractor.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--think_limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--done_from", default="")
    ap.add_argument("--negs", default="./out/hnegs.json",
                    help="kv_hnegs output; LLM-written same-type negatives, preferred over the "
                         "shape-class pool when the item is present (E3 defect C1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    HN = json.load(open(args.negs))["negs"] if args.negs and os.path.exists(args.negs) else {}
    print("[ans] LLM negatives for %d items" % len(HN), flush=True)

    items = load_items(args.data)
    all_items = list(items)
    rnd.shuffle(items)
    items = items[: args.limit]
    by_shape = {}
    for it in all_items:
        by_shape.setdefault(shape_class(it["v"]), []).append(it["v"])
    DONE = set()
    if args.done_from:
        import glob
        for g in args.done_from.split(":"):
            for f in glob.glob(g):
                for line in open(f, errors="ignore"):
                    m = re.match(r"^\[ans\] (\S+) \|", line)
                    if m:
                        DONE.add(m.group(1))

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_t, tail_t = llm.split_wrap()          # thinking enabled
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    HPOS = list(range(H))
    ids = lambda s: list(llm.tok(s, add_special_tokens=False).input_ids)

    def probe(cache, pos, qids):
        _, cand = answer_with_probs(llm, cache, pos, qids, ["A", "B"], ntok=2)
        return cand.get("A", 0.0), cand.get("B", 0.0)

    @torch.no_grad()
    def gen(text, n):
        from transformers import DynamicCache
        o, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), n)
        return o.strip()

    def menu_for(q, gold, cands):
        """negative + order by the no-context prior, kv_e2hot2 verbatim."""
        best = None
        for neg in cands[: args.pool]:
            for ga in (True, False):
                a, b = (gold, neg) if ga else (neg, gold)
                c, pp = assemble(llm, [(H_kv, HPOS)])
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
        # turn-2 readout is a NEW user turn on top of the turn-1 stream
        q2ids = torch.tensor([ids("<|im_end|>\n" + head + recog_q(q, a, b) + tail + "Answer:")],
                             dtype=torch.long, device=llm.device)
        return dict(neg=neg, ga=ga, lo_none=round(lo_none, 4), q2ids=q2ids,
                    plain=llm._ids(recog_q(q, a, b) + tail + "Answer:"))

    def read(cache, pos, m):
        pa, pb = probe(cache, pos, m["q2ids"])
        return round(log_odds(pa, pb, m["ga"]), 4)

    prev_a1 = None   # for the placebo: the previous item's harvested a1 rows
    rows, skip = [], Counter()
    for n_it, it in enumerate(items):
        if it["id"] in DONE:
            continue
        q1, v = it["q"], it["v"]
        order = list(it["titles"]); rnd.shuffle(order)
        ctx = [(t, it["paras"][t]) for t in order]
        body = INSTR1 + "\n\n" + "\n\n".join("[%d] %s: %s" % (i + 1, t, p) for i, (t, p) in enumerate(ctx))
        qtxt = "\n\nQuestion: " + q1
        # ---- stream: head | body | q1 | tail | a1 ----
        b_ids, q_ids, t_ids = ids(body), ids(qtxt), ids(tail)
        prefix_ids = H_ids + b_ids + q_ids + t_ids
        a1 = gen(head + body + qtxt + tail, 64)
        a1_ids = ids(a1)
        if not a1_ids:
            skip["empty_a1"] += 1; continue
        P_b, P_q, P_t, P_a = H, H + len(b_ids), H + len(b_ids) + len(q_ids), len(prefix_ids)
        P_end = P_a + len(a1_ids)
        stream = prefix_ids + a1_ids
        full_kv = encode_block(llm, stream, list(range(P_end)), keep_a=H)
        keep_pos = list(range(P_q, P_t)) + list(range(P_t, P_a)) + list(range(P_a, P_end))   # q1 + tail + a1
        keep_ids = q_ids + t_ids + a1_ids

        def arm(mode, kv=None, kpos=None, kids=None):
            kv = full_kv if kv is None else kv
            kpos = keep_pos if kpos is None else kpos
            kids = keep_ids if kids is None else kids
            if mode == "full":
                rt = torch.tensor(list(range(0, P_end - H)), dtype=torch.long)
                blocks = [(H_kv, HPOS), ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in kv], list(range(H, P_end)))]
            elif mode == "harv":
                rt = torch.tensor([p - H for p in kpos], dtype=torch.long)
                blocks = [(H_kv, HPOS), ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in kv], kpos)]
            else:  # self
                blocks = [(H_kv, HPOS), (encode_block(llm, kids, kpos), kpos)]
            return assemble(llm, blocks)

        # ---- follow-ups ----
        qs = {}
        cands_v = [x for x in by_shape.get(shape_class(v), []) if x.lower() != v.lower()
                   and not wb(x).search(body) and not wb(v).search(x)]
        rnd.shuffle(cands_v)
        llm_negs = [x for x in HN.get(it["id"], {}).get("ans", []) if x.lower() != v.lower()
                    and not wb(x).search(body) and not wb(v).search(x)]
        cands_v = llm_negs + cands_v            # LLM-written first; the probe still picks
        m1 = menu_for(q1, v, list(dict.fromkeys(cands_v)))
        if m1 is not None and wb(v).search(a1):
            qs["stated"] = (q1, v, m1)
        elif m1 is None:
            skip["no_neg_stated"] += 1
        else:
            skip["a1_lacks_gold"] += 1
        # (ii) a year from a distractor passage, absent from a1
        dis = [t for t in order if t not in (it["tA"], it["tB"])]
        rnd.shuffle(dis)
        for tF in dis:
            first = it["paras"][tF].split(". ")[0]
            ys = YEAR.findall(first)
            if len(ys) != 1:
                continue
            f = ys[0]
            if wb(f).search(a1) or wb(f).search(q1):
                continue
            q2 = "In the passages shown earlier, which year is mentioned in the passage about %s?" % strip_paren(tF)
            negs = [str(int(f) + d) for d in (-7, -3, 3, 7, -12, 12, -20, 20)]
            negs = [y for y in negs if not wb(y).search(body)]
            m2 = menu_for(q2, f, negs)
            if m2 is not None:
                qs["unstated"] = (q2, f, m2)
                break
        if not qs:
            skip["no_followup"] += 1
            del full_kv; torch.cuda.empty_cache(); continue

        rec = dict(id=it["id"], q1=q1, v=v, a1=a1, n_a1=len(a1_ids), n_body=len(b_ids),
                   neg_src=("llm" if (qs.get("stated") and qs["stated"][2]["neg"] in llm_negs) else "pool"))
        for kind, (q2, gold, m) in qs.items():
            rec[kind + "_q2"], rec[kind + "_gold"], rec[kind + "_neg"], rec[kind + "_none"] = q2, gold, m["neg"], m["lo_none"]
            c, pp = arm("full");  rec[kind + "_full"] = read(c, pp, m);  del c
            c, pp = arm("harv");  rec[kind + "_ansKV"] = read(c, pp, m); del c
            c, pp = arm("self");  rec[kind + "_ansTX"] = read(c, pp, m); del c
            top = bm25_top(ctx, q2, 3)
            rr = INSTR2 + "\n\n" + "\n\n".join("%s: %s" % (t, p) for t, p in top) + "\n\n"
            c, pp = assemble(llm, [(H_kv, HPOS)])
            pa, pb = probe(c, pp, llm._ids(rr + recog_q(q2, *((gold, m["neg"]) if m["ga"] else (m["neg"], gold))) + tail + "Answer:"))
            rec[kind + "_rerank"] = round(log_odds(pa, pb, m["ga"]), 4); del c
            rec[kind + "_rerank_hasgold"] = int(any(wb(gold).search(p) for _, p in top))
            if prev_a1 is not None:
                pkv, ppos, pids_ = prev_a1
                c, pp = arm("harv", pkv, ppos, pids_); rec[kind + "_p_ansKV"] = read(c, pp, m); del c
            torch.cuda.empty_cache()

        # ---- thinking-mode a1 on a subset ----
        if n_it < args.think_limit:
            a1t = gen(head_t + body + qtxt + tail_t, 400)
            if "</think>" in a1t:
                t_ids2 = ids(tail_t)
                a1t_ids = ids(a1t)
                pre2 = H_ids + b_ids + q_ids + t_ids2
                P_a2 = len(pre2); P_end2 = P_a2 + len(a1t_ids)
                kv2 = encode_block(llm, pre2 + a1t_ids, list(range(P_end2)), keep_a=H)
                kpos2 = list(range(H + len(b_ids), P_end2))          # q1 + tail_t + think + answer
                kids2 = q_ids + t_ids2 + a1t_ids
                ans_part = a1t.split("</think>")[-1].strip()
                rec.update(a1_think=a1t[:2000], n_think=len(a1t_ids), think_answer=ans_part)
                for kind, (q2, gold, m) in qs.items():
                    c, pp = arm("harv", kv2, kpos2, kids2); rec[kind + "_thinkKV"] = read(c, pp, m); del c
                    c, pp = arm("self", kv2, kpos2, kids2); rec[kind + "_thinkTX"] = read(c, pp, m); del c
                    rec[kind + "_think_mentions_gold"] = int(bool(wb(gold).search(a1t)))
                del kv2
            else:
                rec["think_incomplete"] = 1
            torch.cuda.empty_cache()

        # keep this item's harvested q1+a1 rows as the NEXT item's placebo.  arm("harv") index-selects
        # at (pos - H) on a full-width kv, so the rows are placed back into a zero stand-in of that width.
        rt = torch.tensor([p - H for p in keep_pos], dtype=torch.long)
        pw = max(keep_pos) - H + 1
        pkv_full = []
        for (K, V) in full_kv:
            Kf = torch.zeros(K.shape[0], K.shape[1], pw, K.shape[3], dtype=K.dtype, device=K.device)
            Vf = torch.zeros_like(Kf)
            Kf[:, :, rt, :] = K.index_select(2, rt); Vf[:, :, rt, :] = V.index_select(2, rt)
            pkv_full.append((Kf, Vf))
        prev_a1 = (pkv_full, keep_pos, keep_ids)
        del full_kv
        torch.cuda.empty_cache()
        rows.append(rec)
        print("[ans] %s | " % it["id"] + " ".join(
            "%s: none=%+.2f full=%+.2f ansKV=%+.2f ansTX=%+.2f rerank=%+.2f(%d) pKV=%s thinkKV=%s thinkTX=%s"
            % (k, rec[k + "_none"], rec[k + "_full"], rec[k + "_ansKV"], rec[k + "_ansTX"], rec[k + "_rerank"],
               rec[k + "_rerank_hasgold"], ("%+.2f" % rec[k + "_p_ansKV"]) if k + "_p_ansKV" in rec else "na",
               ("%+.2f" % rec[k + "_thinkKV"]) if k + "_thinkKV" in rec else "na",
               ("%+.2f" % rec[k + "_thinkTX"]) if k + "_thinkTX" in rec else "na")
            for k in qs) + " | a1=%r" % a1[:60], flush=True)

    json.dump({"data": args.data, "seed": args.seed, "rows": rows, "skip": dict(skip)}, open(args.out, "w"), indent=1)
    print("[ans] rows=%d skip=%s" % (len(rows), dict(skip)))
    import statistics as st
    for kind in ("stated", "unstated"):
        use = [r for r in rows if kind + "_full" in r]
        if not use:
            continue
        m = lambda k: st.fmean([r[kind + "_" + k] for r in use if kind + "_" + k in r]) if any(kind + "_" + k in r for r in use) else float("nan")
        print("  %-8s n=%3d  none %+6.2f  full %+6.2f  ansKV %+6.2f  ansTX %+6.2f  rerank %+6.2f  p_ansKV %+6.2f  thinkKV %+6.2f  thinkTX %+6.2f"
              % (kind, len(use), m("none"), m("full"), m("ansKV"), m("ansTX"), m("rerank"), m("p_ansKV"), m("thinkKV"), m("thinkTX")))
    print("ANS_DONE", flush=True)


if __name__ == "__main__":
    main()
