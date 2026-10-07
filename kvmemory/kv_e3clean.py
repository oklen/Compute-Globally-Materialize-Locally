"""kv_e3clean.py -- re-read E3's counterfactual control on the items where the counterfactual
actually took.

E3 gate G3 requires zero residual mentions of the gold value once the carrier has been flipped to
the counterfactual, and the run reported 50.  That is not a cosmetic miss.  The whole statistic is
orig - cf, so on a contaminated item the two arms are not different in the way the design assumes:
the model can still read the gold value somewhere in the served rows, which pushes the difference
toward zero.  The headline WAS near zero (+0.0197), so the contaminated items and the null point in
the same direction and must be separated before the null is believed.

This does not re-run anything.  kv_e3ctrl stored cf_gold_residual per item, so the clean subset is
recoverable from the saved json.  Reported here: the full sample, the clean subset, and the
contaminated subset separately.  If the clean subset moves away from zero, the earlier null was an
artifact of contamination; if it does not, the null survives its own worst case and gets stronger.

    PYTHONPATH=. python -m kvmemory.kv_e3clean --res ./out/e3_rep12.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from kvmemory.kv_e2stats import report


ARMS = ("none", "goldev", "p_iso", "p_orig", "p_cf", "q_iso", "q_orig", "q_cf")


def conv_of(r):
    for k in ("conv", "cid", "ds", "conversation"):
        if k in r:
            return r[k]
    return str(r.get("qkey", ""))[:12]


def block(tag, rows, rng):
    if not rows:
        print(f"\n### {tag}: no items")
        return None
    n = len(rows)
    print(f"\n### {tag}   n={n}")
    for a in ARMS:
        pk, pg = f"{a}_pick", f"{a}_p_gold"
        if pk in rows[0]:
            g = sum(int(r.get(pk) == r["gold_letter"]) for r in rows) / n
            m = float(np.mean([r[pg] for r in rows if r.get(pg) is not None]))
            print(f"  {a:8s} gold={g:.3f}  mean p_gold={m:.3f}")

    def dp(r, c):
        return float(r[f"{c}_orig_p_gold"]) - float(r[f"{c}_cf_p_gold"])

    usable = [r for r in rows if r.get("q_orig_p_gold") is not None
              and r.get("q_cf_p_gold") is not None
              and r.get("p_orig_p_gold") is not None
              and r.get("p_cf_p_gold") is not None]
    if not usable:
        print("  (no item has all four p_gold values)")
        return None

    for c in ("q", "p"):
        d = [dp(r, c) for r in usable]
        # follow = the pick tracked whichever value the carrier actually stated
        follow = sum(1 for r in usable
                     if r.get(f"{c}_orig_pick") == r["gold_letter"]
                     and r.get(f"{c}_cf_pick") != r["gold_letter"])
        anti = sum(1 for r in usable
                   if r.get(f"{c}_orig_pick") != r["gold_letter"]
                   and r.get(f"{c}_cf_pick") == r["gold_letter"])
        print(f"  carrier {c}: mean(orig-cf) p_gold = {np.mean(d):+.4f}   follow={follow} anti={anti}")

    by = defaultdict(list)
    for r in usable:
        by[conv_of(r)].append(dp(r, "q") - dp(r, "p"))
    return report(f"E3 PRIMARY (q_dp - p_dp) :: {tag}", by, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="./out/e3_rep12.json")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    R = json.load(open(a.res))
    rows = R["rows"] if isinstance(R, dict) and "rows" in R else R
    key = "cf_gold_residual"
    if key not in rows[0]:
        raise SystemExit(f"{key} absent -- this json predates the residual accounting")

    clean = [r for r in rows if not r[key]]
    dirty = [r for r in rows if r[key]]
    tot = sum(int(r[key]) for r in rows)
    print(f"===== E3 counterfactual, split on gate G3 =====")
    print(f"  {len(rows)} items | clean {len(clean)} | contaminated {len(dirty)} "
          f"({tot} residual gold mentions in total)")
    print("  G3 asks for zero residual mentions; the run reported "
          f"{tot}.  Everything below is that gate, obeyed.")

    for d in sorted({r.get("ds") for r in rows}):
        sub = [r for r in rows if r.get("ds") == d]
        print(f"  corpus {d}: {len(sub)} items, {len(sub) - sum(1 for r in sub if r[key])} clean")

    out = {}
    out["all"] = block("ALL ITEMS (what the run printed)", rows, rng)
    out["clean"] = block("CLEAN -- counterfactual actually took (G3 satisfied)", clean, rng)
    out["dirty"] = block("CONTAMINATED -- gold still visible after the flip", dirty, rng)

    for d in sorted({r.get("ds") for r in rows}):
        sub = [r for r in clean if r.get("ds") == d]
        if len(sub) >= 20:
            out[f"clean_{d}"] = block(f"CLEAN, corpus={d} (reported separately, never merged)", sub, rng)

    if out["clean"] and out["all"]:
        c, al = out["clean"]["point"], out["all"]["point"]
        print(f"\n  clean {c:+.4f}  vs  all {al:+.4f}   (shift {c-al:+.4f})")
        if out["clean"]["p_wcr"] > 0.05:
            print("  The counterfactual sensitivity stays indistinguishable from zero on the items")
            print("  where the flip actually took.  The null is NOT a contamination artifact -- it")
            print("  survives its own worst case, and E3's read of the carrier is that flipping the")
            print("  value it states barely moves the answer.")
        else:
            print("  The clean subset separates from zero: the earlier near-zero was diluted by")
            print("  items where the gold value was still visible after the flip.  G3 is therefore")
            print("  load-bearing and the full-sample number must not be quoted.")


if __name__ == "__main__":
    main()
