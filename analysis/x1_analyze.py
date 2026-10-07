"""X1 readout: on in-window AMA-Bench questions, is what the KV carries from unselected context information or influence?

Usage: python3 x1_analyze.py ans_*.jsonl
Arms: tx re-read the text / kv gathered KV / iso spans encoded separately / p3 perturb 3 unselected upstream steps / pall replace every unselected upstream step with neutral filler.
Readouts (episode-clustered bootstrap, 2000 draws): per-arm accuracy; kv-tx, kv-iso, p3-kv, pall-kv;
among changed answers, how many go right->wrong vs wrong->right (sign test).
Information hypothesis: pall < kv, changes skewed right->wrong; influence hypothesis: accuracy unchanged, changes symmetric.
"""
import collections
import json
import math
import random
import sys

rows = []
for p in sys.argv[1:]:
    for line in open(p):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows = [r for r in rows if r.get("j_p3") is not None]
eps = sorted({r["episode_id"] for r in rows})
by_ep = collections.defaultdict(list)
for r in rows:
    by_ep[r["episode_id"]].append(r)
print(f"n={len(rows)} QA, {len(eps)} episodes, domains {dict(collections.Counter(r['domain'] for r in rows))}")


def acc(rs, k):
    return sum(r[k] for r in rs) / max(1, len(rs))


for k in ("j_tx", "j_kv", "j_iso", "j_p3", "j_pall"):
    print(f"  {k[2:]:5s} acc {acc(rows, k):.3f}")


def boot(diff, B=2000, seed=0):
    rng = random.Random(seed)
    est = diff(rows)
    vals = []
    for _ in range(B):
        sample = [r for e in (rng.choice(eps) for _ in eps) for r in by_ep[e]]
        vals.append(diff(sample))
    vals.sort()
    return est, vals[int(.025 * B)], vals[int(.975 * B) - 1]


for name, a, b in (("kv - tx", "j_kv", "j_tx"), ("kv - iso", "j_kv", "j_iso"),
                   ("p3 - kv", "j_p3", "j_kv"), ("pall - kv", "j_pall", "j_kv")):
    est, lo, hi = boot(lambda rs, a=a, b=b: acc(rs, a) - acc(rs, b))
    print(f"  {name:10s} {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


def sign_p(k, n):
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


for arm in ("p3", "pall"):
    ch = [r for r in rows if r[f"chg_{arm}"]]
    rw = sum(1 for r in ch if r["j_kv"] == 1 and r[f"j_{arm}"] == 0)
    wr = sum(1 for r in ch if r["j_kv"] == 0 and r[f"j_{arm}"] == 1)
    print(f"  {arm}: answer changed {len(ch)}/{len(rows)} ({len(ch)/len(rows):.0%}); first token changed {sum(r[f'flip_{arm}'] for r in rows)}/{len(rows)}; "
          f"of the changes, right->wrong {rw}, wrong->right {wr} (sign test p={sign_p(rw, rw + wr):.2f}), the other {len(ch) - rw - wr} keep their correctness")
for d in sorted({r["domain"] for r in rows}):
    rs = [r for r in rows if r["domain"] == d]
    print(f"  [{d}] n={len(rs)} tx {acc(rs,'j_tx'):.3f} kv {acc(rs,'j_kv'):.3f} iso {acc(rs,'j_iso'):.3f} p3 {acc(rs,'j_p3'):.3f} pall {acc(rs,'j_pall'):.3f}")
