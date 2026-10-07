"""x3_readout.py -- the model's OWN readout on exactly the kv_payprobe items, menu form.

Free generation on these items never reads the value (the model answers with the source register's
NAME: "M mirrors S" -> "S"), so it cannot be compared with the probe. This uses the paper's menu
protocol instead: list the candidate values as lettered options (independent decoys for num3), read
the next-token distribution over the letters, take the argmax. Three serving arms per item:
  harv  [H] + carrier rows harvested from the prefill with the source present; source omitted
  full  [H; prefix; carrier] all served (ceiling)
  iso   [H] + carrier encoded with no prefix (floor: chance)
Items are regenerated with the SAME rng sequence as kv_payprobe (seed 0, n=512, payload order), so
accuracy lines up item-for-item with the probe.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        python -m kvmemory.x3_readout 512 ./out/x3/readout_qwen3.json      # n items, output path
"""
from __future__ import annotations

import json
import random
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_phantom import HDR, ids_of
from kvmemory.kv_payprobe import PAYLOADS, make_items, texts

LET = "ABCDEFGHIJ"


@torch.no_grad()
def next_logits(llm, blocks, new_ids):
    cache, pos = assemble(llm, blocks)
    cache_len = pos.shape[0]
    L = new_ids.shape[1]
    nxt = int(pos.max().item()) + 1
    out = llm.model(input_ids=new_ids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(nxt, nxt + L, device=llm.device).unsqueeze(0),
                    cache_position=torch.arange(cache_len, cache_len + L, device=llm.device),
                    attention_mask=torch.ones(1, cache_len + L, dtype=torch.long, device=llm.device))
    lg = out.logits[0, -1].float()
    del cache
    return lg


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    outp = sys.argv[2] if len(sys.argv) > 2 else "./out/x3/readout_qwen3.json"
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = ids_of(llm, head + HDR)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    let_ids = [llm.tok.encode(c, add_special_tokens=False)[0] for c in LET]
    rng = random.Random(0)        # same sequence as kv_payprobe
    rng2 = random.Random(1)       # option order / decoys
    res = {}
    for kind in PAYLOADS:
        items = make_items(n, kind, rng)
        ok = {"harv": 0, "full": 0, "iso": 0}
        ok_gt = {"harv": 0, "full": 0, "iso": 0}
        k_used = None
        for it in items:
            pre, car, _ = texts(it)
            if kind == "num3":
                pool = set()
                while len(pool) < 9:
                    d = str(rng2.randrange(100, 1000))
                    if d != it["v"]:
                        pool.add(d)
                opts = list(pool) + [it["v"]]
            else:
                opts = list(PAYLOADS[kind])
            rng2.shuffle(opts)
            k_used = len(opts)
            gold = opts.index(it["v"])
            menu = " ".join(f"{LET[i]}) {o}" for i, o in enumerate(opts))
            q = (f"\n\nQuestion: What is the value of register {it['M']}? Options: {menu}. "
                 f"Answer with the letter only." + tail)
            qids = llm._ids(q)
            p_ids, c_ids = ids_of(llm, pre), ids_of(llm, car)
            P, C = len(p_ids), len(c_ids)
            est, een = H + P, H + P + C
            allpos = list(range(H + P)) + list(range(est, een))
            hk = encode_block(llm, H_ids + p_ids + c_ids, allpos, keep_a=H + P)
            fk = encode_block(llm, H_ids + p_ids + c_ids, allpos, keep_a=H)
            ik = encode_block(llm, H_ids + c_ids, list(range(H)) + list(range(est, een)), keep_a=H)
            arms = {"harv": [(H_kv, list(range(H))), (hk, list(range(est, een)))],
                    "full": [(H_kv, list(range(H))), (fk, list(range(H, een)))],
                    "iso": [(H_kv, list(range(H))), (ik, list(range(est, een)))]}
            for a, blocks in arms.items():
                lg = next_logits(llm, blocks, qids)
                pick = int(torch.stack([lg[t] for t in let_ids[:k_used]]).argmax())
                ok[a] += int(pick == gold)
            if kind == "num3":   # coarse readout: above 500 or not (two options, order shuffled)
                o2 = ["above 500", "500 or below"]
                rng2.shuffle(o2)
                g2 = o2.index("above 500" if int(it["v"]) > 500 else "500 or below")
                q2 = (f"\n\nQuestion: Is the value of register {it['M']} above 500? Options: A) {o2[0]} B) {o2[1]}. "
                      f"Answer with the letter only." + tail)
                q2ids = llm._ids(q2)
                for a, blocks in arms.items():
                    lg = next_logits(llm, blocks, q2ids)
                    ok_gt[a] += int(int(torch.stack([lg[let_ids[0]], lg[let_ids[1]]]).argmax()) == g2)
        res[kind] = {"k": k_used, "n": n, **{f"menu_{a}": ok[a] / n for a in ok}}
        if kind == "num3":
            res[kind].update({f"gt500_{a}": ok_gt[a] / n for a in ok_gt})
        print(kind, json.dumps(res[kind]), flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("X3R_DONE", flush=True)


if __name__ == "__main__":
    main()
