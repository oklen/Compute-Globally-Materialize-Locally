"""kv_swgate.py -- sliding-window oracle gate for compact-cache serving (Gemma-4).

Question: does the compact-cache readout (_greedy_pos over concatenated retained rows at
contiguous cache slots) reproduce NATIVE serving of the same rows at their ORIGINAL positions
under the model's native attention (incl. sliding-window local layers) + a key-mask that drops
the omitted rows?

Both arms use the SAME retained-row KV, sliced from one full prefill -- so the ONLY difference is
the query's attention geometry:
  COMPACT : retained rows at contiguous slots 0..k-1; query at slot k. A local layer's window is
            measured in COMPACT slots, so a far row (few slots away) is always in-window.
  NATIVE  : retained rows kept at their ORIGINAL positions in the full cache; dropped rows blocked
            by a 2D key-mask; the query at the original next position. A local layer's window is
            measured in ORIGINAL positions, so a retained row whose original position is >window
            from the query is masked out (visible only in global layers).

Item: a FAR root ("register M mirrors register S", source S set to a donor value, source OMITTED)
placed early, then long filler, then a donor-invariant NEAR row, then the query. On a full-attn
model NATIVE==COMPACT. On Gemma-4, if the far row's donor is followed MORE under COMPACT than
NATIVE, the compact cache over-exposes it -> the concern is real. A SHORT control (far row within
one window) must show NATIVE==COMPACT.

    SPRAG_MODEL_PATH=/path/to/Gemma-4-12B SPRAG_ATTN_IMPL=sdpa SPRAG_ENCODE_CHUNK=4096 \
      PYTHONPATH=/home/tiger CUDA_VISIBLE_DEVICES=0 tf5env/bin/python -m kvmemory.kv_swgate \
      --items 24 --gap 2400 --seed 1
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
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble, slice_kv
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]


def pick_last(o):
    u = o.upper()
    a, b = u.rfind("ONLINE"), u.rfind("OFFLINE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "OFFLINE"
    if b < 0:
        return "ONLINE"
    return "ONLINE" if a > b else "OFFLINE"


@torch.no_grad()
def _decode(llm, cache, first_logits, first_next, base_pos, base_cache, attn_builder, gen):
    """Greedy-decode `gen` tokens after an already-ingested query. attn_builder(total)->(1,total)."""
    dev = llm.device
    toks, nxt, cur, cur_pos = [], first_next, base_cache, base_pos
    for _ in range(gen):
        if nxt in llm.eos_ids:
            break
        toks.append(nxt)
        out = llm.model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache,
                        use_cache=True, position_ids=torch.tensor([[cur_pos]], device=dev),
                        cache_position=torch.tensor([cur], device=dev),
                        attention_mask=attn_builder(cur + 1))
        nxt = int(out.logits[0, -1].argmax())
        cur += 1
        cur_pos += 1
    return llm.tok.decode(toks, skip_special_tokens=True).strip()


@torch.no_grad()
def serve_native(llm, full_kv, L, retained_pos, qids, gen):
    """Full cache (all L rows) + 2D key-mask(retained=1) + query at original pos L; native window."""
    dev = llm.device
    cache = DynamicCache()
    for li, (K, V) in enumerate(full_kv):
        cache.update(K.to(dev), V.to(dev), li)
    Lq = qids.shape[1]
    keym = torch.zeros(1, L, dtype=torch.long, device=dev)
    keym[0, torch.tensor(sorted(retained_pos), device=dev)] = 1

    def attn_builder(total):
        return torch.cat([keym, torch.ones(1, total - L, dtype=torch.long, device=dev)], dim=1)

    qpos = torch.arange(L, L + Lq, device=dev)
    out = llm.model(input_ids=qids, past_key_values=cache,
                    use_cache=True, position_ids=qpos.unsqueeze(0), cache_position=qpos,
                    attention_mask=attn_builder(L + Lq))
    fl = out.logits[0, -1].float()
    ans = _decode(llm, cache, fl, int(fl.argmax()), L + Lq, L + Lq, attn_builder, gen)
    return fl, ans


@torch.no_grad()
def serve_compact(llm, full_kv, spans, qids, gen):
    """Slice the retained spans from the SAME full_kv, assemble to a compact cache, _greedy_pos."""
    blocks = [(slice_kv(full_kv, a, b), list(range(a, b))) for (a, b) in spans]
    cache, pos = assemble(llm, blocks)
    dev = llm.device
    clen = pos.shape[0]
    Lq = qids.shape[1]
    nxt_pos = int(pos.max().item()) + 1
    qpos = torch.arange(nxt_pos, nxt_pos + Lq, device=dev)
    cpos = torch.arange(clen, clen + Lq, device=dev)
    out = llm.model(input_ids=qids, past_key_values=cache,
                    use_cache=True, position_ids=qpos.unsqueeze(0), cache_position=cpos,
                    attention_mask=torch.ones(1, clen + Lq, dtype=torch.long, device=dev))
    fl = out.logits[0, -1].float()

    def attn_builder(total):
        return torch.ones(1, total, dtype=torch.long, device=dev)
    ans = _decode(llm, cache, fl, int(fl.argmax()), nxt_pos + Lq, clen + Lq, attn_builder, gen)
    return fl, ans


def build_item(llm, rng, H_ids, H, gap):
    """Direct-value probe: a FAR row states M's value (= donor); a donor-invariant decoy sits
    NEAR the query, with `gap` tokens of fixed filler between. All non-donor content is drawn ONCE
    so the two donor variants are byte-identical except the far row's value token."""
    D, M, Dn = rng.sample(NAMES, 3)
    others = [n for n in NAMES if n not in (D, M, Dn)]
    decoy_val = rng.choice(STATUS2)
    near_val = rng.choice(STATUS2)
    nfill = max(1, gap // 20)
    fillers = [f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
               for _ in range(nfill)]
    ev = lambda i, body: f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n"
    far_body = lambda v: f"register {M} set to state {v}"
    # position budget: pad the far event so ONLINE/OFFLINE occupy identical token length
    NL = llm.tok("\n", add_special_tokens=False).input_ids[-1]
    tlen = lambda v: len(llm.tok(ev(1, far_body(v)), add_special_tokens=False).input_ids)
    far_budget = max(tlen("ONLINE"), tlen("OFFLINE"))

    def make(v):
        seq = ([f"register {D} set to state {decoy_val}", far_body(v)]
               + fillers + [f"register {Dn} set to state {near_val}"])
        ids, pos, spans = list(H_ids), list(range(H)), {}
        for idx, b in enumerate(seq):
            tid = list(llm.tok(ev(idx, b), add_special_tokens=False).input_ids)
            if idx == 1:
                while len(tid) < far_budget:
                    tid.append(NL)          # pad the far row to a fixed budget
            spans[idx] = (len(ids), len(ids) + len(tid))
            ids += tid
            pos += list(range(spans[idx][0], spans[idx][1]))
        return ids, pos, spans[1], spans[len(seq) - 1], len(seq)
    return D, M, Dn, make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--gap", type=int, default=2400)   # ~tokens between far root and near row
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--menu", action="store_true", default=True)
    ap.add_argument("--out", default="./out/swgate.json")
    args = ap.parse_args()
    llm = HFBackend()
    print("MODEL_CLASS", type(llm.model).__name__, flush=True)
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)

    def cand_ids():
        for pre in (" ", ""):
            a = llm.tok(pre + "ONLINE", add_special_tokens=False).input_ids
            b = llm.tok(pre + "OFFLINE", add_special_tokens=False).input_ids
            if a[0] != b[0]:
                return a[0], b[0]
        a = llm.tok("ONLINE", add_special_tokens=False).input_ids
        b = llm.tok("OFFLINE", add_special_tokens=False).input_ids
        return a[0], b[0]
    on_id, off_id = cand_ids()
    print("CANDIDATE_TOKENS on=%d off=%d" % (on_id, off_id), flush=True)

    def readout(fl):  # menu / candidate-logit (no abstention), the Gemma-4 protocol
        return "ONLINE" if float(fl[on_id]) > float(fl[off_id]) else "OFFLINE"

    rng = random.Random(args.seed)

    rows = []
    nat = {"follow": 0, "anti": 0, "const": 0, "other": 0}
    cmp = {"follow": 0, "anti": 0, "const": 0, "other": 0}
    argmatch = 0
    devs = []
    for it in range(args.items):
        D, M, Dn, make = build_item(llm, rng, H_ids, H, args.gap)
        q = (f"\n\nQuestion: What state is register {M} in? Answer ONLINE or OFFLINE:")
        qids = llm._ids(q + tail)
        # byte-identity guard: the two donor variants must match except inside the far row
        ion, pon, fs, ns, _ = make("ONLINE")
        iof, _, _, _, _ = make("OFFLINE")
        if len(ion) != len(iof) or ion[:fs[0]] != iof[:fs[0]] or ion[fs[1]:] != iof[fs[1]:]:
            print(f"[sw] it{it} SKIP (donor not byte-identical outside far row)", flush=True)
            continue
        na, ca = {}, {}
        gap_far = None
        for v in STATUS2:
            ids, pos, far_span, near_span, nseq = make(v)
            L = len(ids)
            gap_far = L - far_span[0]  # query-to-far distance (>window => far masked natively)
            full_kv = encode_block(llm, ids, pos)  # all rows
            retained = list(range(*far_span)) + list(range(*near_span))
            fl_n, ans_n = serve_native(llm, full_kv, L, retained, qids, args.gen)
            fl_c, ans_c = serve_compact(llm, full_kv, [far_span, near_span], qids, args.gen)
            na[v], ca[v] = readout(fl_n), readout(fl_c)
            if v == "ONLINE":
                argmatch += int(int(fl_n.argmax()) == int(fl_c.argmax()))
                devs.append(float((fl_n - fl_c).abs().max()))
            del full_kv
            torch.cuda.empty_cache()

        def cls(a):
            ro, rf = a["ONLINE"], a["OFFLINE"]
            if ro == rf:
                return "const" if ro in ("ONLINE", "OFFLINE") else "other"
            if (ro, rf) == ("ONLINE", "OFFLINE"):
                return "follow"
            if (ro, rf) == ("OFFLINE", "ONLINE"):
                return "anti"
            return "other"
        nat[cls(na)] += 1
        cmp[cls(ca)] += 1
        rows.append({"it": it, "gap_far": gap_far, "native": na, "compact": ca})
        print(f"[sw] it{it} gap~{gap_far} | NATIVE {na['ONLINE']}/{na['OFFLINE']} "
              f"({cls(na)}) | COMPACT {ca['ONLINE']}/{ca['OFFLINE']} ({cls(ca)})", flush=True)

    n = len(rows)
    json.dump({"gap": args.gap, "n": n, "native": nat, "compact": cmp,
               "first_argmatch": argmatch, "maxdev_mean": sum(devs) / max(1, len(devs)),
               "rows": rows}, open(args.out, "w"), indent=1)
    print("\n===== SLIDING-WINDOW ORACLE (gap=%d) =====" % args.gap)
    print(f"far-root donor-follow:  NATIVE follow/anti/const/other = "
          f"{nat['follow']}/{nat['anti']}/{nat['const']}/{nat['other']}")
    print(f"                        COMPACT follow/anti/const/other = "
          f"{cmp['follow']}/{cmp['anti']}/{cmp['const']}/{cmp['other']}")
    print(f"first-token argmax NATIVE==COMPACT: {argmatch}/{n}; "
          f"max|dlogit| mean={sum(devs)/max(1,len(devs)):.3f}")
    verdict = ("COMPACT over-exposes far row (concern REAL)"
               if cmp["follow"] - nat["follow"] >= max(3, n // 8) else
               "NATIVE ~ COMPACT (concern benign at this gap)")
    print("VERDICT:", verdict)
    print("SWGATE_DONE", flush=True)


if __name__ == "__main__":
    main()
