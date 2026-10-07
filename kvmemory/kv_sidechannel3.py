"""kv_sidechannel3.py -- = kv_sidechannel2 (10-02), plus: an episode that runs out of GPU memory is skipped and
logged instead of killing the worker (Qwen3-32B leaves ~14 GB beside its weights on an 80 GB card).

kv_sidechannel2.py -- AMA follow-up to kv_sidechannel (the 66% answer-change result): is what the
gathered KV carries from UNSELECTED upstream turns information, or diffuse influence?

kv_sidechannel showed that perturbing unselected upstream turns changes the gathered-KV answer on 66%
of queries (111/168) while a text re-prefill is invariant by construction. It recorded only WHETHER
answers changed, never whether they were right. This run judges every answer, so a change can be
signed (right->wrong vs wrong->right), and adds the strongest perturbation (all unselected context).

  tx       re-prefill header + kept spans (verbatim) + q             compact positions, joint
  kv       prefill the full trajectory once, gather kept spans' KV at original positions, decode
  iso      kept spans encoded alone (block-diagonal mask, header shared) at original positions
  kv_p3    as kv, but 3 unselected upstream turns replaced, length-matched, by another step's text
           (the kv_sidechannel perturbation, replicated with the same rule)
  kv_pall  as kv, but EVERY unselected upstream turn replaced, length-matched, by neutral filler
           (all unselected context destroyed; kept spans' text identical, positions ~preserved)

Reads (paired per QA; cluster by episode at analysis time):
  acc(kv) - acc(tx)        what the in-situ KV is worth over the same text
  acc(kv_pall) - acc(kv)   does destroying ALL unselected context cost accuracy?
  among answers that changed under p3 / pall: right->wrong vs wrong->right
Information predicts kv_pall < kv and changes skewed right->wrong; influence predicts no accuracy
cost and symmetric changes.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_sidechannel2 --shard 0 --queue_dir ./out/x1_q --ep_mod 2 --ep_rem 0 \
        --max_tokens 16000 --out ./out/x1/sum_s0.json --ans_out ./out/x1/ans_s0.jsonl
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
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge, norm
from kvmemory.kv_select_smoke import split_wrap_nothink, first_logits_masked_full
from kvmemory.kv_scope import isolated_cache, hop_bucket
from kvmemory.kv_sidechannel import length_matched_replacement

FILLER = "the agent then observed a different unrelated state and continued. "


@torch.no_grad()
def gathered_answer(llm, texts, header, kept, qtext, qids, ans_tokens):
    """Prefill `texts` once, gather the kept spans' KV at their positions, decode. Returns
    (answer, first-token argmax)."""
    cache, spans, _ = llm.prefill_full(texts, header)
    hlen = spans[0][0]
    l1 = llm.first_logits_subselect(cache, spans, hlen, kept, qtext)
    sub, kpos = llm.subselect_cache(cache, spans, hlen, kept)
    ans, _, _ = llm._greedy_pos(sub, kpos, qids, ans_tokens)
    del cache, sub
    torch.cuda.empty_cache()
    return ans, int(l1.argmax())


@torch.no_grad()
def process_episode(llm, ep, head, tail, router, args, do_gate):
    segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
    n_seg = len(segment_texts)
    header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
              f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
    full_cache, spans, total = llm.prefill_full(segment_texts, header)
    header_len = spans[0][0]
    hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
    old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
    id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}

    rows, gate = [], []
    for qi, qa in enumerate(ep.qa[:args.max_qa]):
        q = qa["question"]
        gold = qa.get("answer", "") or ""
        picked = router.select(q, old, args.k)
        kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
        kept_set = set(kept)
        qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
        qids = llm._ids(qtext)
        upstream = [i for i in range(n_seg) if i not in kept_set and i < max(kept)]

        # tx / kv / iso  (identical to kv_scope)
        text = header + "".join(segment_texts[i] for i in kept) + qtext
        ans_tx, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
        l0 = llm.first_logits_subselect(full_cache, spans, header_len, kept, qtext)
        sub, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
        ans_kv, _, _ = llm._greedy_pos(sub, kpos, qids, args.ans_tokens)
        del sub
        iso_c, ipos = isolated_cache(llm, header, segment_texts, kept, spans, header_len)
        ans_iso, _, _ = llm._greedy_pos(iso_c, ipos, qids, args.ans_tokens)
        del iso_c
        top0 = int(l0.argmax())
        if do_gate and len(gate) < 3:  # gathered KV must equal full-prefill-then-mask (faithfulness)
            masked = first_logits_masked_full(llm, header, segment_texts, kept, qtext)
            gate.append(int(top0 == int(masked.argmax())))

        # kv_p3: the kv_sidechannel rule, seeded per (episode, qa) so every worker agrees
        rng = random.Random(ep.episode_id * 1009 + qi * 7 + args.seed)
        ans_p3 = ans_pall = None
        top_p3 = top_pall = None
        n_p3 = 0
        if upstream:
            targets = rng.sample(upstream, min(args.n_perturb, len(upstream)))
            n_p3 = len(targets)
            p3 = list(segment_texts)
            for i in targets:
                donor = segment_texts[(i + n_seg // 2 + 1) % n_seg]
                p3[i] = length_matched_replacement(segment_texts[i], donor)
            ans_p3, top_p3 = gathered_answer(llm, p3, header, kept, qtext, qids, args.ans_tokens)
            pall = list(segment_texts)
            for i in upstream:
                pall[i] = length_matched_replacement(segment_texts[i], FILLER)
            ans_pall, top_pall = gathered_answer(llm, pall, header, kept, qtext, qids, args.ans_tokens)

        def J(a):
            return None if a is None else int(judge(llm, head, tail, q, gold, a))

        r = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qa.get("type", "?"),
             "hop": hop_bucket(q), "qi": qi, "n_seg": n_seg, "tok": total, "n_kept": len(kept),
             "n_upstream": len(upstream), "n_p3": n_p3, "q": q, "gold": gold,
             "ans_tx": ans_tx, "ans_kv": ans_kv, "ans_iso": ans_iso, "ans_p3": ans_p3, "ans_pall": ans_pall,
             "j_tx": J(ans_tx), "j_kv": J(ans_kv), "j_iso": J(ans_iso), "j_p3": J(ans_p3), "j_pall": J(ans_pall),
             "chg_p3": None if ans_p3 is None else int(norm(ans_p3) != norm(ans_kv)),
             "chg_pall": None if ans_pall is None else int(norm(ans_pall) != norm(ans_kv)),
             "flip_p3": None if top_p3 is None else int(top_p3 != top0),
             "flip_pall": None if top_pall is None else int(top_pall != top0),
             "same_kv_tx": int(norm(ans_kv) == norm(ans_tx))}
        rows.append(r)
    del full_cache
    torch.cuda.empty_cache()
    return rows, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_ep", type=int, default=1000)
    ap.add_argument("--max_qa", type=int, default=100)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_perturb", type=int, default=3)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ep_mod", type=int, default=1, help="static split across machines: keep eps with idx%%mod==rem")
    ap.add_argument("--ep_rem", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--smoke", type=int, default=0, help="if >0: first episode only, this many QA")
    ap.add_argument("--out", default="./out/x1/sum.json")
    ap.add_argument("--ans_out", default="./out/x1/ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    alleps = load_episodes(args.data, max_tokens=args.max_tokens)
    bydom = defaultdict(list)
    for e in alleps:
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())
    eps = [e for i, e in enumerate(eps) if i % args.ep_mod == args.ep_rem]
    nqa = sum(min(len(e.qa), args.max_qa) for e in eps)
    print(f"[w{args.shard}] {len(eps)} episodes (<= {args.max_tokens} tok, split {args.ep_rem}/{args.ep_mod}), "
          f"{nqa} QA, {len(bydom)} domains, k={args.k} hot={args.hot}", flush=True)
    if args.smoke:
        eps = eps[:1]
        args.max_qa = args.smoke

    os.makedirs(os.path.dirname(args.ans_out) or ".", exist_ok=True)
    ansf = open(args.ans_out, "a", encoding="utf-8")
    agg = {"n": 0, "tx": 0, "kv": 0, "iso": 0, "np": 0, "p3": 0, "pall": 0, "chg_p3": 0, "chg_pall": 0,
           "gate_ok": 0, "gate_n": 0, "eps": 0}

    def fold(ep, rows, gate):
        agg["eps"] += 1
        agg["gate_ok"] += sum(gate); agg["gate_n"] += len(gate)
        for r in rows:
            agg["n"] += 1
            agg["tx"] += r["j_tx"]; agg["kv"] += r["j_kv"]; agg["iso"] += r["j_iso"]
            if r["j_p3"] is not None:
                agg["np"] += 1; agg["p3"] += r["j_p3"]; agg["pall"] += r["j_pall"]
                agg["chg_p3"] += r["chg_p3"]; agg["chg_pall"] += r["chg_pall"]
            ansf.write(json.dumps(r, ensure_ascii=False) + "\n")
        ansf.flush()
        json.dump({"shard": args.shard, **agg}, open(args.out, "w"), indent=2)
        n = max(1, agg["n"]); m = max(1, agg["np"])
        print(f"[w{args.shard}] ep {ep.episode_id} {ep.domain} | n={agg['n']} tx {agg['tx']/n:.3f} kv {agg['kv']/n:.3f} "
              f"iso {agg['iso']/n:.3f} | p3 {agg['p3']/m:.3f} pall {agg['pall']/m:.3f} "
              f"| chg p3 {agg['chg_p3']}/{agg['np']} pall {agg['chg_pall']}/{agg['np']} | gate {agg['gate_ok']}/{agg['gate_n']}",
              flush=True)

    first = True
    if args.queue_dir and not args.smoke:
        os.makedirs(args.queue_dir, exist_ok=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                rows, gate = process_episode(llm, eps[i], head, tail, router, args, do_gate=first)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[w{args.shard}] OOM skip ep {eps[i].episode_id}", flush=True)
                continue
            first = False
            fold(eps[i], rows, gate)
    else:
        for ep in eps:
            rows, gate = process_episode(llm, ep, head, tail, router, args, do_gate=first)
            first = False
            fold(ep, rows, gate)
    ansf.close()
    print(f"X1B_DONE shard={args.shard} n={agg['n']}", flush=True)


if __name__ == "__main__":
    main()
