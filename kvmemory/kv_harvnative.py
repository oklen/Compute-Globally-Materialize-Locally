"""kv_harvnative.py -- native vs compact serving for the real-dialog harvest arm (Gemma-4).

App. D showed our compact serving re-packs retained rows at contiguous cache slots, so on
Gemma-4's sliding-window layers the local window is measured in COMPACT slots rather than original
positions. Table 6's Gemma-4 cells therefore report the compact-cache implementation, not native
serving. This script serves the SAME items both ways and pairs them per QA.

  compact : slice retained rows -> assemble -> query at max(retained)+1   (the deployed path)
  native  : keep the FULL cache at original positions + a 2D key-mask that drops the omitted
            rows -> query at original position L; the model's own sliding-window mask applies.

Both arms read out identically (first-token logit of " A" vs " B"), so native-vs-compact is not
confounded by the readout. The compact arm is ALSO read the paper's way (8-token greedy + parse)
to bridge to the published numbers.

The isolated control is served natively by writing the isolated-encoded carrier KV into its own
original slots of the full cache, serving, then restoring -- so harv/iso differ only in the KV
content at the carrier slots, never in mask geometry or query position.

    SPRAG_MODEL_PATH=/path/to/Gemma-4-12B SPRAG_ATTN_IMPL=sdpa SPRAG_ENCODE_CHUNK=4096 \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_harvnative --ds locomo --out ./out/hn.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_harvest3 import load_realtalk, load_locomo, select_qa
from kvmemory.kv_harvest4 import recog_q, hard_neg, pick_ab


@torch.no_grad()
def ab_logit(llm, fl, a_id, b_id):
    return "A" if float(fl[a_id]) > float(fl[b_id]) else "B"


@torch.no_grad()
def serve_native(llm, full_gpu, L, keep_pos, qids):
    """Full cache at ORIGINAL positions + key-mask(keep_pos)=1; query at original position L.
    use_cache=False so the 65k cache is never concatenated/copied (one forward, logits only)."""
    dev = llm.device
    cache = DynamicCache()
    for li, (K, V) in enumerate(full_gpu):
        cache.update(K, V, li)
    Lq = qids.shape[1]
    keym = torch.zeros(1, L, dtype=torch.long, device=dev)
    keym[0, torch.tensor(sorted(keep_pos), dtype=torch.long, device=dev)] = 1
    attn = torch.cat([keym, torch.ones(1, Lq, dtype=torch.long, device=dev)], dim=1)
    qpos = torch.arange(L, L + Lq, device=dev)
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=False,
                    position_ids=qpos.unsqueeze(0), cache_position=qpos,
                    attention_mask=attn)
    fl = out.logits[0, -1].float()
    del cache
    return fl


@torch.no_grad()
def serve_compact(llm, blocks, qids, gen=0):
    """blocks: [(kv_slices, positions)] ascending -> assemble, query at max(pos)+1."""
    cache, pos = assemble(llm, blocks)
    dev = llm.device
    clen = pos.shape[0]
    Lq = qids.shape[1]
    nxt = int(pos.max().item()) + 1
    qpos = torch.arange(nxt, nxt + Lq, device=dev)
    cpos = torch.arange(clen, clen + Lq, device=dev)
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=qpos.unsqueeze(0), cache_position=cpos,
                    attention_mask=torch.ones(1, clen + Lq, dtype=torch.long, device=dev))
    fl = out.logits[0, -1].float()
    txt = None
    if gen:
        toks, nx, cur, cp = [], int(fl.argmax()), clen + Lq, nxt + Lq
        for _ in range(gen):
            if nx in llm.eos_ids:
                break
            toks.append(nx)
            o2 = llm.model(input_ids=torch.tensor([[nx]], device=dev), past_key_values=cache,
                           use_cache=True, position_ids=torch.tensor([[cp]], device=dev),
                           cache_position=torch.tensor([cur], device=dev),
                           attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev))
            nx = int(o2.logits[0, -1].argmax())
            cur += 1
            cp += 1
        txt = llm.tok.decode(toks, skip_special_tokens=True).strip()
    del cache
    return fl, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["realtalk", "locomo", "all"], required=True)
    ap.add_argument("--rt_path", default="./data/REALTALK")
    ap.add_argument("--lc_path", default="./data/locomo10.json")
    ap.add_argument("--max_conv", type=int, default=0)
    ap.add_argument("--conv_idx", type=int, default=-1,
                    help="process ONLY this conversation index; run one per process so a 65k "
                         "cache is fully reclaimed between conversations (avoids fragmentation OOM)")
    ap.add_argument("--max_qa", type=int, default=0)
    ap.add_argument("--out", default="./out/hn.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    H = len(H_ids)

    # candidate token ids for the first-token A/B readout
    a_id = llm.tok(" A", add_special_tokens=False).input_ids[-1]
    b_id = llm.tok(" B", add_special_tokens=False).input_ids[-1]
    print(f"[hn] A/B token ids: {a_id} {b_id}", flush=True)

    if args.ds == "all":
        tasks = ([("realtalk", c) for c in load_realtalk(args.rt_path)]
                 + [("locomo", c) for c in load_locomo(args.lc_path)])
    else:
        loader = load_realtalk(args.rt_path) if args.ds == "realtalk" else load_locomo(args.lc_path)
        tasks = [(args.ds, c) for c in loader]
    if args.max_conv:
        tasks = tasks[: args.max_conv]
    if args.conv_idx >= 0:
        if args.conv_idx >= len(tasks):
            print(f"[hn] conv_idx {args.conv_idx} >= {len(tasks)}; nothing to do")
            print("HN_DONE")
            return
        tasks = [tasks[args.conv_idx]]

    rows = []
    for ds_name, cv in tasks:
        qas = select_qa(cv)
        if args.max_qa:
            qas = qas[: args.max_qa]
        if not qas:
            continue
        turns = cv["turns"]

        # full prefill INCLUDING the header rows (no keep_a) -> positions 0..L-1
        eids = [list(llm.tok(t + "\n", add_special_tokens=False).input_ids) for _, _, t in turns]
        spans, cur = [], H
        for e in eids:
            spans.append((cur, cur + len(e)))
            cur += len(e)
        flat = [t for e in eids for t in e]
        L = cur
        full_cpu = encode_block(llm, H_ids + flat, list(range(L)))
        full_gpu = [(K.to(llm.device, non_blocking=True), V.to(llm.device, non_blocking=True))
                    for K, V in full_cpu]
        del full_cpu
        torch.cuda.empty_cache()
        print(f"[hn] {cv['cid']} turns={len(turns)} qa={len(qas)} L={L} "
              f"kv={sum(K.numel()*K.element_size()*2 for K,_ in full_gpu)/2**30:.1f}GiB", flush=True)

        for qa in qas:
            neg = hard_neg(qas, qa)
            if neg is None:
                continue
            gold = qa["gold"].strip()
            gold_is_a = int(hashlib.sha1(qa["q"].encode()).hexdigest(), 16) % 2 == 0
            a, b = (gold, neg) if gold_is_a else (neg, gold)
            gold_letter = "A" if gold_is_a else "B"
            qids = llm._ids(recog_q(qa["q"], a, b) + tail)
            rec = {"ds": ds_name, "conv": cv["cid"], "cat": qa["cat"], "gold": gold,
                   "neg": neg, "gold_letter": gold_letter, "L": L}

            carrier_pos = [pp for i in qa["carriers"] for pp in range(*spans[i])]
            keep_pos = list(range(H)) + carrier_pos

            # ---- isolated carrier KV (same encode as the paper's iso arm) ----
            iso_parts = []
            for i in qa["carriers"]:
                aa, bb = spans[i]
                iso_parts.append((encode_block(llm, eids[i], list(range(aa, bb))),
                                  list(range(aa, bb))))

            # ---- COMPACT arms (deployed path) ----
            hdr = [(K[:, :, :H], V[:, :, :H]) for K, V in full_gpu]
            sel = torch.tensor(carrier_pos, dtype=torch.long, device=llm.device)
            harv_sub = [(K.index_select(2, sel), V.index_select(2, sel)) for K, V in full_gpu]
            for nm, blocks in (("harv", [(hdr, list(range(H))), (harv_sub, carrier_pos)]),
                               ("iso", [(hdr, list(range(H)))] + iso_parts)):
                fl, t8 = serve_compact(llm, blocks, qids, gen=8)     # the published budget
                _, t64 = serve_compact(llm, blocks, qids, gen=64)    # a budget big enough to answer
                rec[f"{nm}_compact"] = int(ab_logit(llm, fl, a_id, b_id) == gold_letter)
                rec[f"{nm}_gen8"] = int(pick_ab(t8) == gold_letter)
                rec[f"{nm}_gen8_parsed"] = int(pick_ab(t8) is not None)
                rec[f"{nm}_gen64"] = int(pick_ab(t64) == gold_letter)
                rec[f"{nm}_gen64_parsed"] = int(pick_ab(t64) is not None)
            del harv_sub

            # ---- NATIVE arms (original positions + key-mask) ----
            fl = serve_native(llm, full_gpu, L, keep_pos, qids)
            rec["harv_native"] = int(ab_logit(llm, fl, a_id, b_id) == gold_letter)

            saved = []
            for kv_i, ppl in iso_parts:
                aa, bb = ppl[0], ppl[-1] + 1
                saved.append((aa, bb, [(K[:, :, aa:bb].clone(), V[:, :, aa:bb].clone())
                                       for K, V in full_gpu]))
                for li, (K, V) in enumerate(full_gpu):
                    K[:, :, aa:bb] = kv_i[li][0].to(llm.device)
                    V[:, :, aa:bb] = kv_i[li][1].to(llm.device)
            fl = serve_native(llm, full_gpu, L, keep_pos, qids)
            rec["iso_native"] = int(ab_logit(llm, fl, a_id, b_id) == gold_letter)
            for aa, bb, orig in saved:                      # restore the harvested rows
                for li, (K, V) in enumerate(full_gpu):
                    K[:, :, aa:bb] = orig[li][0]
                    V[:, :, aa:bb] = orig[li][1]

            del iso_parts, saved
            rows.append(rec)
            torch.cuda.empty_cache()

        del full_gpu, hdr
        gc.collect()
        torch.cuda.empty_cache()
        n = sum(1 for r in rows if r["conv"] == cv["cid"])
        print(f"[hn] {cv['cid']} done n={n}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)   # checkpoint every conversation

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    if rows:
        print(f"\n===== {args.ds} n={len(rows)} =====")
        for k in ("harv_gen8", "harv_gen8_parsed", "iso_gen8", "iso_gen8_parsed",
                  "harv_gen64", "harv_gen64_parsed", "iso_gen64", "iso_gen64_parsed",
                  "harv_compact", "iso_compact", "harv_native", "iso_native"):
            print(f"  {k:20s} {sum(r[k] for r in rows)/len(rows):.3f}")
        d = lambda a, b: sum(r[a] - r[b] for r in rows) / len(rows)
        print(f"\n  harv-iso  gen8(published budget)={d('harv_gen8','iso_gen8'):+.3f}"
              f"  gen64={d('harv_gen64','iso_gen64'):+.3f}"
              f"  compact-logit={d('harv_compact','iso_compact'):+.3f}"
              f"  native-logit={d('harv_native','iso_native'):+.3f}")
    print("HN_DONE")


if __name__ == "__main__":
    main()
