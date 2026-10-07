"""Cluster-bootstrap power for E2's PRIMARY (harv - iso), on the ACTUAL serve-set structure.

Why this is re-run and not reused: the first estimate was made against the BM25 dry run, 601 items
over per-conversation sizes [23..78].  The real sets are 1269 items over [53..168] -- the loose
paraphrase channel admitted 1,786 extra V rows, so far fewer items are skipped for `no value row`.
More items per conversation shrink WITHIN-conversation noise, but the cluster bootstrap resamples
CONVERSATIONS, and there are still only ten of them.  Between-conversation variance is therefore
the binding constraint, and doubling n buys much less than the sample count suggests.  Reporting
the new number matters either way: it is what decides how much an E2 null is allowed to claim.

Outcome model.  Each item is scored under both arms, so the paired difference is ternary:
+1 (harv right, iso wrong), -1 (iso right, harv wrong), 0 (agree).  `m` is the DISCORDANCE rate --
the fraction of items where the two arms disagree at all -- and `a` the share of those favouring
harv, so delta = m*(2a-1).  Conversation-level heterogeneity enters as jitter on a, which is where
real between-topic variation lives (some conversations are simply more harvest-friendly).

    PYTHONPATH=. python -m kvmemory.kv_e2power      # CPU only; --sizes defaults to the ten E2 clusters
"""
import argparse, numpy as np


def simulate(sizes, delta, m, tau, B, S, rng, alpha=0.05):
    """Return (power, mean CI width). tau = sd of the conversation-level shift in delta."""
    sizes = np.asarray(sizes)
    K = len(sizes)
    hits = 0
    widths = []
    for _ in range(S):
        # per-conversation delta, clipped so |delta_i| never exceeds the discordance rate
        di = np.clip(delta + rng.normal(0, tau, K), -m, m)
        a = (di / m + 1) / 2                                  # share of discordant favouring harv
        cols = []
        for k in range(K):
            u = rng.random(sizes[k])
            x = np.where(u < m * a[k], 1.0, np.where(u < m, -1.0, 0.0))
            cols.append(x)
        means = np.array([c.mean() for c in cols])
        w = sizes / sizes.sum()
        # cluster bootstrap: resample CONVERSATIONS with replacement, item-weighted mean
        idx = rng.integers(0, K, size=(B, K))
        bm = means[idx]
        bw = w[idx]
        boot = (bm * bw).sum(1) / bw.sum(1)
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        widths.append(hi - lo)
        if lo > 0 or hi < 0:
            hits += 1
    return hits / S, float(np.mean(widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="53,101,112,127,130,130,133,147,168,168")
    ap.add_argument("--old-sizes", default="23,41,56,60,63,64,70,71,75,78")
    ap.add_argument("--m", type=float, default=0.20, help="discordance rate")
    ap.add_argument("--tau", type=float, default=0.02, help="between-conversation sd of delta")
    ap.add_argument("-B", type=int, default=1500)
    ap.add_argument("-S", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    new = [int(x) for x in a.sizes.split(",")]
    old = [int(x) for x in a.old_sizes.split(",")]
    rng = np.random.default_rng(a.seed)

    print(f"discordance m={a.m}  between-conv sd tau={a.tau}  B={a.B} sims={a.S} seed={a.seed}")
    print(f"new sets: n={sum(new)} over {len(new)} conversations {new}")
    print(f"old (bm25 dry run): n={sum(old)} over {len(old)} conversations\n")
    print(f"{'delta':>7} | {'power(new)':>10} {'CIwidth':>8} | {'power(old)':>10} {'CIwidth':>8}")
    print("-" * 56)
    for d in (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
        pn, wn = simulate(new, d, a.m, a.tau, a.B, a.S, np.random.default_rng(a.seed))
        po, wo = simulate(old, d, a.m, a.tau, a.B, a.S, np.random.default_rng(a.seed))
        tag = "   <- false positive rate" if d == 0 else ""
        print(f"{d:7.2f} | {pn:10.3f} {wn:8.4f} | {po:10.3f} {wo:8.4f}{tag}")
    print("\nSensitivity to the between-conversation sd (delta = 0.02, the effect E3 measured):")
    for tau in (0.00, 0.01, 0.02, 0.04):
        pn, wn = simulate(new, 0.02, a.m, tau, a.B, a.S, np.random.default_rng(a.seed))
        print(f"  tau={tau:.2f} -> power {pn:.3f}  CI width {wn:.4f}")


if __name__ == "__main__":
    main()
