"""kv_e2stats.py -- the interval for E2's PRIMARY, which kv_e2.py does not compute.

kv_e2.py prints point estimates only.  The pre-registration says "conversation clustered
bootstrap, B=4000" and that is what this file's `percentile` method does -- but LoCoMo has
only TEN conversations, and a percentile cluster bootstrap on ten clusters was measured
(kv_e2power.py) to reject a true null 9.5-10.7% of the time at a nominal 5%.  Reporting a
"significant" harv-iso from that interval would be reporting an instrument defect.

So the PRIMARY interval here is the wild cluster bootstrap-t with Rademacher weights and the
null imposed (Cameron-Gelbach-Miller), which is the standard remedy for few clusters.  With
G=10 there are only 2^10 = 1024 possible sign vectors, so they are ENUMERATED rather than
sampled: the p-value is exact up to the discreteness of the reference distribution, with no
Monte Carlo error at all.  (The floor on a two-sided p is 2/1024 = 0.002.)

Everything is closed-form in the per-cluster sums, which is why the enumeration is cheap.
For sign vector s and null value d0, with A_g = S_g - n_g*d0 (S_g = sum of the paired
difference in cluster g):

    dhat* - d0    = (1/n) * sum_g s_g A_g                        =: B
    sum_i u*_gi   = s_g A_g - n_g B
    V*            = c * sum_g (s_g A_g - n_g B)^2 / n^2 ,  c = G/(G-1)

so no item-level array is ever touched inside the loop.

The CI is obtained by inverting the test over a grid of d0 -- the set of nulls the data does
not reject.  All three methods are printed side by side, on purpose: the gap between them is
itself a reportable fact about how much the ten-cluster design can support.

Run --selfcheck FIRST and read it before trusting any number below it.  It measures each
method's actual size under a true null and its power under a known planted effect, on this
experiment's real cluster sizes.  A method that cannot reject a planted effect is a dead
instrument, and a "no significant difference" from a dead instrument is not evidence.

    PYTHONPATH=. python -m kvmemory.kv_e2stats --selfcheck
    PYTHONPATH=. python -m kvmemory.kv_e2stats --res ./out/e2_full.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------- core statistics

def cluster_sums(diffs_by_conv):
    """-> (S, ng) as float arrays, one entry per conversation."""
    keys = sorted(diffs_by_conv)
    S = np.array([float(np.sum(diffs_by_conv[k])) for k in keys])
    ng = np.array([float(len(diffs_by_conv[k])) for k in keys])
    return S, ng


def crve(S, ng, dhat):
    """Cluster-robust SE of the pooled mean.  c = G/(G-1) * (N-1)/(N-K) with K=1."""
    G, n = len(S), ng.sum()
    c = G / (G - 1.0)
    V = c * np.sum((S - ng * dhat) ** 2) / n ** 2
    return float(np.sqrt(V))


def _sign_matrix(G, rng, cap=16):
    """All 2^G Rademacher vectors when that is small enough, else a random sample.
    Enumeration is exact; the sampled branch is only a fallback for other corpora."""
    if G <= cap:
        idx = np.arange(2 ** G)
        s = 1.0 - 2.0 * ((idx[:, None] >> np.arange(G)[None, :]) & 1).astype(float)
        return s, True
    return rng.choice([-1.0, 1.0], size=(9999, G)), False


def wcr_p(S, ng, d0, signs, dhat, se):
    """Wild cluster bootstrap-t p-value for H0: delta == d0, null imposed."""
    n = ng.sum()
    A = S - ng * d0                                   # restricted cluster residual sums
    B = signs @ A / n                                 # (M,)  = dhat* - d0
    resid = signs * A[None, :] - ng[None, :] * B[:, None]
    G = len(S)
    V = (G / (G - 1.0)) * np.sum(resid ** 2, axis=1) / n ** 2
    ok = V > 0
    if not ok.any():
        # Every cluster's paired difference is identically d0, so the statistic has no dispersion
        # and there is nothing to reject.  This must NOT come back as nan: a gate written as
        # `p > 0.05` reads nan as False, i.e. as SIGNIFICANT, which is the exact wrong direction.
        # Caught by the dry run on a placebo arm whose difference was zero on every item.
        return 1.0 if abs(dhat - d0) < 1e-12 else 0.0
    t_star = np.abs(B[ok]) / np.sqrt(V[ok])
    t_obs = (abs(dhat - d0) / se) if se > 0 else (0.0 if abs(dhat - d0) < 1e-12 else np.inf)
    return float(np.mean(t_star >= t_obs))


def wcr_ci(S, ng, signs, dhat, se, alpha=0.05, half=8.0, grid=481):
    """Invert the WCR test: the CI is the set of nulls not rejected."""
    if se <= 0:
        return (float(dhat), float(dhat))     # degenerate: a point, not a nan
    d0s = np.linspace(dhat - half * se, dhat + half * se, grid)
    keep = [d for d in d0s if wcr_p(S, ng, d, signs, dhat, se) > alpha]
    return (float(min(keep)), float(max(keep))) if keep else (float("nan"), float("nan"))


def percentile_ci(S, ng, rng, B=4000, alpha=0.05):
    """The PRE-REGISTERED interval: resample conversations with replacement.  Kept so the
    paper can show it next to the calibrated one, not because it is trustworthy at G=10."""
    G = len(S)
    idx = rng.integers(0, G, size=(B, G))
    boot = (S[idx].sum(1)) / (ng[idx].sum(1))
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def t_ci(dhat, se, G, alpha=0.05):
    """Cluster-robust t interval with G-1 degrees of freedom -- the cheap middle option."""
    try:
        from scipy import stats
        crit = stats.t.ppf(1 - alpha / 2, G - 1)
    except Exception:                       # scipy-free fallback: t_{.975} table
        crit = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                10: 2.228, 11: 2.201, 12: 2.179, 19: 2.093}.get(G - 1, 1.96)
    return dhat - crit * se, dhat + crit * se


def report(name, diffs_by_conv, rng, alpha=0.05, quiet=False):
    S, ng = cluster_sums(diffs_by_conv)
    G, n = len(S), ng.sum()
    # A single cluster is not a small sample, it is NO sample: every interval here is built from
    # between-cluster variation, and crve's c = G/(G-1) divides by zero at G=1. Say so plainly
    # rather than crash (or, worse, return a number that looks like an estimate). Same rule as
    # the nan-gate: a degenerate input must produce an unmistakable statement, not an exception
    # and not a plausible-looking value.
    if G < 2 or n == 0:
        if not quiet:
            print("\n  %s   n=%d  clusters=%d" % (name, int(n), G))
            print("    NOT ESTIMABLE: %d cluster(s). Every interval reported here is built from "
                  "between-conversation variation, so at least 2 are required." % G)
        return dict(name=name, n=int(n), clusters=G, point=float("nan"), cluster_mean=float("nan"),
                    se=float("nan"), p_wcr=float("nan"), ci_wcr=[float("nan")] * 2,
                    ci_t=[float("nan")] * 2, ci_percentile=[float("nan")] * 2, not_estimable=True)
    dhat = S.sum() / n
    se = crve(S, ng, dhat)
    signs, exact = _sign_matrix(G, rng)
    p_wcr = wcr_p(S, ng, 0.0, signs, dhat, se)
    ci_w = wcr_ci(S, ng, signs, dhat, se, alpha)
    ci_p = percentile_ci(S, ng, rng, alpha=alpha)
    ci_t = t_ci(dhat, se, G, alpha)
    # each conversation weighted equally, as a sensitivity check: cluster sizes run 53..168
    cm = float(np.mean(S / ng))
    if not quiet:
        print(f"\n  {name}   n={int(n)}  clusters={G}")
        if se <= 0:
            print(f"    DEGENERATE: every conversation's paired difference is identically "
                  f"{dhat:+.4f}; no dispersion, so no test is possible (p reported as 1.0).")
        print(f"    point (pooled)      {dhat:+.4f}      cluster-mean {cm:+.4f}")
        print(f"    cluster-robust SE   {se:.4f}")
        print(f"    WCR p (null imposed, {'all ' + str(len(signs)) + ' sign vectors' if exact else 'sampled'}) = {p_wcr:.4f}")
        print(f"    PRIMARY  wild cluster bootstrap-t 95% CI  [{ci_w[0]:+.4f}, {ci_w[1]:+.4f}]")
        print(f"    (ref)    cluster-t, {G-1} df              [{ci_t[0]:+.4f}, {ci_t[1]:+.4f}]")
        print(f"    (ref)    percentile bootstrap, prereg     [{ci_p[0]:+.4f}, {ci_p[1]:+.4f}]  <- size ~0.10, see --selfcheck")
    return dict(name=name, n=int(n), clusters=G, point=dhat, cluster_mean=cm, se=se,
                p_wcr=p_wcr, ci_wcr=ci_w, ci_t=ci_t, ci_percentile=ci_p)


# ---------------------------------------------------------------- instrument self-check

def _simulate(sizes, delta, m, tau, rng):
    """One synthetic experiment with the same cluster structure: paired difference is ternary
    (+1 / 0 / -1), discordance rate m, and the conversation-level share favouring harv jitters
    by tau.  Same generator as kv_e2power.py so the two files cannot drift apart."""
    out = {}
    di = np.clip(delta + rng.normal(0, tau, len(sizes)), -m, m)
    a = (di / m + 1) / 2
    for g, (sz, ag) in enumerate(zip(sizes, a)):
        disc = rng.random(sz) < m
        sign = np.where(rng.random(sz) < ag, 1.0, -1.0)
        out[g] = disc * sign
    return out


def selfcheck(sizes, reps=400, m=0.20, tau=0.02, seed=1234):
    """Measure what each interval actually does on THIS cluster structure.

    Two things must both hold before any E2 number is quoted:
      size   -- under a true null, rejection should be ~5%.  The percentile bootstrap is
                expected to fail this (it was measured at 9.5-10.7%); WCR should pass.
      power  -- under a PLANTED effect the method must reject.  This is the injected-fault
                test: without it, a null result is indistinguishable from a dead instrument.
    """
    rng = np.random.default_rng(seed)
    print(f"  cluster sizes {list(map(int, sizes))}  (G={len(sizes)}, n={int(sum(sizes))})")
    print(f"  {reps} synthetic experiments per row, m={m} tau={tau}\n")
    print(f"  {'delta':>7} | {'WCR':>18} | {'cluster-t':>18} | {'percentile (prereg)':>20}")
    print("  " + "-" * 72)
    rows = {}
    for delta in (0.00, 0.02, 0.05, 0.10):
        rej = {"wcr": 0, "t": 0, "pct": 0}
        for _ in range(reps):
            d = _simulate(sizes, delta, m, tau, rng)
            S, ng = cluster_sums(d)
            dhat = S.sum() / ng.sum()
            se = crve(S, ng, dhat)
            signs, _ = _sign_matrix(len(S), rng)
            if se > 0 and wcr_p(S, ng, 0.0, signs, dhat, se) <= 0.05:
                rej["wcr"] += 1
            lo, hi = t_ci(dhat, se, len(S))
            if lo > 0 or hi < 0:
                rej["t"] += 1
            lo, hi = percentile_ci(S, ng, rng, B=1000)
            if lo > 0 or hi < 0:
                rej["pct"] += 1
        r = {k: v / reps for k, v in rej.items()}
        rows[delta] = r
        tag = "  <- SIZE (target 0.05)" if delta == 0 else ""
        print(f"  {delta:>7.2f} | {r['wcr']:>18.3f} | {r['t']:>18.3f} | {r['pct']:>20.3f}{tag}")
    print()
    size = rows[0.00]
    verdict = []
    verdict.append(("WCR size", size["wcr"], size["wcr"] <= 0.075))
    verdict.append(("cluster-t size", size["t"], size["t"] <= 0.075))
    verdict.append(("percentile size", size["pct"], size["pct"] <= 0.075))
    verdict.append(("WCR power @ planted delta=0.10", rows[0.10]["wcr"], rows[0.10]["wcr"] >= 0.50))
    for nm, val, ok in verdict:
        print(f"    [{'PASS' if ok else 'FAIL'}] {nm} = {val:.3f}")
    print("\n  Reading: the percentile row is the pre-registered interval.  If its size is well")
    print("  above 0.05 it must not be used to call significance, and the WCR column is PRIMARY.")
    print("  Power at delta=0.02 is low by construction at G=10 -- an E2 null is therefore WEAK")
    print("  evidence of no effect, and must be written that way (plan row E2-d).")
    return rows


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="", help="kv_e2.py --out json")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sets", default="./out/e2_sets.json",
                    help="only used by --selfcheck, to read the real cluster sizes")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.selfcheck:
        try:
            items = json.load(open(a.sets))["items"]
            sizes = list(np.bincount([int(i["conv"]) for i in items]))
            sizes = [s for s in sizes if s]
        except Exception as e:
            print(f"  (could not read {a.sets}: {e}; using the recorded E2 serve-set sizes)")
            sizes = [53, 101, 112, 127, 130, 130, 133, 147, 168, 168]
        print("===== instrument self-check =====")
        selfcheck(sizes, reps=a.reps, seed=a.seed)
        if not a.res:
            return

    R = json.load(open(a.res))
    rows = R["rows"]
    print(f"\n===== E2 statistics (arm={R.get('arm')}, k={R.get('k')}, n={len(rows)}) =====")

    for a_ in ("none", "text", "iso", "harv", "p_iso", "p_harv"):
        k = f"j_{a_}"
        if rows and k in rows[0]:
            print(f"  {a_:7s} {sum(r[k] for r in rows) / len(rows):.3f}")

    def by_conv(f):
        d = defaultdict(list)
        for r in rows:
            d[r["conv"]].append(float(f(r)))
        return d

    out = {}
    out["primary"] = report("PRIMARY  harv - iso", by_conv(lambda r: r["j_harv"] - r["j_iso"]), rng)
    out["placebo"] = report("CONTROL  p_harv - p_iso", by_conv(lambda r: r["j_p_harv"] - r["j_p_iso"]), rng)
    out["text_iso"] = report("(ref)    text - iso", by_conv(lambda r: r["j_text"] - r["j_iso"]), rng)

    # the placebo is a gate, not a decoration: state the reading explicitly
    import math
    pp, pr = out["placebo"]["p_wcr"], out["primary"]["p_wcr"]
    if not (math.isfinite(pp) and math.isfinite(pr)):
        print(f"\n  gate: NOT EVALUABLE -- a p-value came back non-finite "
              f"(primary {pr}, placebo {pp}). Fix that before reading anything below.")
        pp, pr = 1.0, 1.0
    p_ok = pp > 0.05
    pr_ok = pr <= 0.05
    print("\n  gate: " + (
        "PRIMARY significant and placebo not -- the effect is specific to this question's "
        "omitted value (plan row E2-a)." if pr_ok and p_ok else
        "PRIMARY significant BUT SO IS THE PLACEBO -- read as generic contextualisation, "
        "not materialisation (plan row E2-b)." if pr_ok and not p_ok else
        "PRIMARY not significant.  Given the measured power at G=10 this is WEAK evidence of "
        "no effect, not evidence of absence (plan row E2-d)."))

    # three-cell decomposition; only wrong->gold is unambiguous
    cells = defaultdict(int)
    for r in rows:
        cells[(r["c_iso"], r["c_harv"])] += 1
    print("\n  three-cell decomposition (iso -> harv), discordants only:")
    for k in sorted(cells):
        if k[0] != k[1]:
            print(f"    {k[0]:7s} -> {k[1]:7s} : {cells[k]}")
    w2g = cells[("wrong", "gold")]
    g2w = cells[("gold", "wrong")]
    a2g = cells[("abstain", "gold")]
    g2a = cells[("gold", "abstain")]
    print(f"\n    unambiguous state transfer   wrong->gold {w2g}  vs  gold->wrong {g2w}   net {w2g-g2w:+d}")
    print(f"    readout activation only      abstain->gold {a2g}  vs  gold->abstain {g2a}   net {a2g-g2a:+d}")
    if (w2g - g2w) <= 0 < (a2g - g2a):
        print("    -> the gain is entirely abstention conversion: report as READOUT ACTIVATION,")
        print("       not as a transferred state (plan row E2-c; this is the X8 Gemma-4 pattern).")
    if w2g + g2w:
        out["wrong_to_gold"] = report("(cell)   wrong->gold vs gold->wrong",
                                      by_conv(lambda r: float(r["c_iso"] == "wrong" and r["c_harv"] == "gold")
                                              - float(r["c_iso"] == "gold" and r["c_harv"] == "wrong")), rng)

    # floor check: a strong prior compresses every paired difference (plan row E2-f)
    if rows and "j_none" in rows[0]:
        nf = [r for r in rows if not r["j_none"]]
        print(f"\n  floor: none={sum(r['j_none'] for r in rows)/len(rows):.3f}"
              f"   ({len(nf)}/{len(rows)} items the model cannot already answer)")
        if len(nf) >= 30:
            d = defaultdict(list)
            for r in nf:
                d[r["conv"]].append(float(r["j_harv"] - r["j_iso"]))
            out["primary_none_fail"] = report("PRIMARY on none-fail subset", d, rng)

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
