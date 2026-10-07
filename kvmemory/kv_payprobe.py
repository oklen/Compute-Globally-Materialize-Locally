"""kv_payprobe.py -- is the capacity envelope a WRITE limit or a READ limit?

Materialize's access envelope (binary .934, four/eight-way near chance, three-digit 0/192) is an
output-interface measurement; the paper names a representation-level probe as the outstanding
adjudicator (Limitations, L1092-1094). This run is that probe.

Items (synthetic, Qwen3-8B):  [H] <step 1> set_register(S, v) ... <step 2> filler ... <step 3>
"register M mirrors register S."   v drawn uniformly from the payload's value set:
    bin  2 values | k4  4 values | k8  8 values | num3  100..999 (probed per digit and as >500)

Per item, one forward over [H; prefix; carrier] (harv) and one over [H; carrier] at the same
positions with no prefix (iso, the floor: the carrier text never names v). From each forward take
the carrier span's hidden states (mean and last token) at every 4th layer; also the source value's
own tokens from the harv forward (the probe's positive control). A linear probe (PCA-128 + L2
multinomial logistic regression, 4-fold CV grouped by register-name pair) decodes v.
Generation readout on the same items: serve [H] + carrier rows harvested from the harv prefill,
prefix omitted, ask for M's value, exact match.

Reads: probe(harv) vs probe(iso) vs generation, per payload. If k4/k8 decode well above chance from
harv while generation sits at chance, the envelope is a read limit (written but not readable).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_payprobe --n 512 --out ./out/x3/probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block
from kvmemory.kv_phantom import HDR, FILLERS, ids_of, serve_and_answer

PAYLOADS = {
    "bin": ["ONLINE", "OFFLINE"],
    "k4": ["RED", "GREEN", "BLUE", "YELLOW"],
    "k8": ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "ORANGE", "BLACK", "WHITE"],
    "num3": None,
}
NAMES = ["ALPHA", "BRAVO", "CEDAR", "DELTA", "EMBER", "FALCON", "GARNET", "HARBOR", "IVORY", "JASPER",
         "KESTREL", "LUMEN", "MAPLE", "NOVA", "ONYX", "PRISM", "QUARTZ", "RAVEN", "SIERRA", "TUNDRA"]


def make_items(n, kind, rng):
    pairs = [(a, b) for a in NAMES for b in NAMES if a != b]
    items = []
    for i in range(n):
        S, M = pairs[rng.randrange(len(pairs))]
        if kind == "num3":
            v = str(rng.randrange(100, 1000))
        else:
            vals = PAYLOADS[kind]
            v = vals[i % len(vals)]          # balanced classes
        items.append({"S": S, "M": M, "v": v, "note": FILLERS[i % len(FILLERS)].format(obj="register")})
    rng.shuffle(items)
    return items


def texts(it):
    pre = (f"<step 1>\naction: set_register('{it['S']}', '{it['v']}')\nobservation: register {it['S']} "
           f"= {it['v']}\n<step 2>\naction: log_note('{it['note']}')\nobservation: ok\n")
    car = f"<step 3>\naction: link_registers()\nobservation: register {it['M']} mirrors register {it['S']}.\n"
    q = f"\n\nQuestion: What is the value of register {it['M']}? Answer with the value only:"
    return pre, car, q


@torch.no_grad()
def hidden(llm, ids, positions, spans, layers):
    """Core forward (no lm_head). Returns {span_name: [n_layers, 2*d]} with mean and last token."""
    core = getattr(llm.model, "model", llm.model)
    core = getattr(core, "language_model", None) or core
    dev = llm.device
    out = core(input_ids=torch.tensor([ids], device=dev), position_ids=torch.tensor([positions], device=dev),
               output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    res = {}
    for name, (a, b) in spans.items():
        res[name] = torch.stack([torch.cat([hs[l][0, a:b].float().mean(0), hs[l][0, b - 1].float()])
                                 for l in layers]).half().cpu()
    return res


def probe_cv(X, y, groups, k, folds=4, dim=128, lam=1e-3, seed=0):
    """Grouped k-fold: PCA (fit on train) -> L2 multinomial logistic regression (LBFGS). Mean test acc."""
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    accs = []
    for f in range(folds):
        test_g = set(g[f::folds])
        te = torch.tensor([gi in test_g for gi in groups])
        Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        q = min(dim, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, V = torch.pca_lowrank(Xtr, q=q, center=False)
        Ztr, Zte = Xtr @ V, Xte @ V
        W = torch.zeros(q, k, requires_grad=True)
        b = torch.zeros(k, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Ztr @ W + b, ytr) + lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        accs.append(float(((Zte @ W + b).argmax(1) == yte).float().mean()))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--layer_step", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="./out/x3/probe.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    H_ids = ids_of(llm, head + HDR)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    nl = llm.model.config.num_hidden_layers
    layers = list(range(args.layer_step, nl + 1, args.layer_step))
    n = args.smoke or args.n
    result = {"layers": layers, "n": n, "payloads": {}}

    for kind in PAYLOADS:
        items = make_items(n, kind, rng)
        feats = {"harv": [], "iso": [], "src": []}
        gen_ok = 0
        for it in items:
            pre, car, q = texts(it)
            p_ids, c_ids = ids_of(llm, pre), ids_of(llm, car)
            P, C = len(p_ids), len(c_ids)
            est, een = H + P, H + P + C
            # where the value sits inside the prefix (positive control span)
            vs = None
            for cand in (f" {it['v']}", it["v"]):
                vtok = ids_of(llm, cand)
                vs = next((j for j in range(P - len(vtok), -1, -1) if p_ids[j:j + len(vtok)] == vtok), None)
                if vs is not None:
                    break
            spans = {"harv": (est, een)}
            if vs is not None:
                spans["src"] = (H + vs, H + vs + len(vtok))
            hv = hidden(llm, H_ids + p_ids + c_ids, list(range(een)), spans, layers)
            hi = hidden(llm, H_ids + c_ids, list(range(H)) + list(range(est, een)), {"iso": (H, H + C)}, layers)
            feats["harv"].append(hv["harv"]); feats["iso"].append(hi["iso"])
            feats["src"].append(hv.get("src", torch.zeros_like(hv["harv"])))
            # generation readout: carrier rows harvested, prefix omitted
            allpos = list(range(H + P)) + list(range(est, een))
            hk = encode_block(llm, H_ids + p_ids + c_ids, allpos, keep_a=H + P)
            out = serve_and_answer(llm, [(H_kv, list(range(H))), (hk, list(range(est, een)))], q, tail, ntok=8)
            m = re.search(r"[A-Za-z]+|\d+", out or "")
            ans = m.group(0).upper() if m else ""
            gen_ok += int(ans == it["v"].upper())
        groups = [f"{it['S']}-{it['M']}" for it in items]
        if kind == "num3":
            targets = {"hundreds": [int(it["v"][0]) - 1 for it in items],   # 1..9 -> 0..8
                       "tens": [int(it["v"][1]) for it in items],
                       "ones": [int(it["v"][2]) for it in items],
                       "gt500": [int(int(it["v"]) > 500) for it in items]}
        else:
            vals = PAYLOADS[kind]
            targets = {"value": [vals.index(it["v"]) for it in items]}
        pr = {"gen_exact": gen_ok / n, "chance": {}, "probe": {}}
        for tname, ylist in targets.items():
            k = max(ylist) + 1
            y = torch.tensor(ylist)
            pr["chance"][tname] = max(ylist.count(c) for c in set(ylist)) / n
            pr["probe"][tname] = {}
            for src in ("harv", "iso", "src"):
                Xall = torch.stack(feats[src]).float()      # [n, n_layers, 2d]
                pr["probe"][tname][src] = [round(probe_cv(Xall[:, li], y, groups, k), 4) for li in range(len(layers))]
            print(f"[{kind}/{tname}] chance {pr['chance'][tname]:.3f} gen {pr['gen_exact']:.3f} | "
                  f"harv max {max(pr['probe'][tname]['harv']):.3f} iso max {max(pr['probe'][tname]['iso']):.3f} "
                  f"src max {max(pr['probe'][tname]['src']):.3f}", flush=True)
        result["payloads"][kind] = pr
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
    print(f"X3_DONE n={n}", flush=True)


if __name__ == "__main__":
    main()
