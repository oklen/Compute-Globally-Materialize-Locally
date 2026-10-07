"""ansres.py -- read-out of the answer-KV session-memory experiment (kv_ans.py).

Two follow-up kinds, log-odds (gold vs neg, forced choice):
    stated    q2 asks a fact a1 STATED       (narrow form; ansKV should ~ ansTX ~ full)
    unstated  q2 asks a fact in D1 a1 did NOT state (broad form; law predicts ansKV ~ none, rerank wins)

Arms: full (keep whole turn-1 KV) / ansKV (harvest a1 rows, drop D1) / ansTX (re-encode a1 rows) /
      thinkKV (keep reasoning too) / rerank (re-retrieve top-3 for q2) / none / p_ansKV (placebo).
The deployable question: after answering, can we EVICT the passages and keep only the answer's KV?
"""
import json, glob, sys, statistics as st, collections
sys.path.insert(0, '.')
try:
    from wcrfast import W
except Exception:
    W = None

J = None
for cand in ('ans.json',):
    if glob.glob(cand):
        J = json.load(open(glob.glob(cand)[0])); break
if J is None:
    print('no ans.json'); sys.exit(1)
rows = J['rows']
print('n=%d  skip=%s' % (len(rows), J.get('skip')))
# cluster = none (each item independent); use item-level bootstrap via a synthetic single cluster? 
# kv_ans has no conversation clustering (each HotpotQA item is independent) -> plain paired stats.
import math
def paired(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return float('nan'), float('nan'), 0
    m = st.fmean(vals); sd = st.pstdev(vals) or 1e-9
    se = sd / math.sqrt(len(vals))
    t = m / se
    # normal approx two-sided p
    p = math.erfc(abs(t) / math.sqrt(2))
    return m, p, len(vals)

ARMS = ("none", "full", "ansKV", "ansTX", "thinkKV", "thinkTX", "rerank", "p_ansKV")
for kind in ("stated", "unstated"):
    use = [r for r in rows if kind + "_full" in r]
    if not use:
        continue
    print('\n' + '=' * 74 + '\n%s  (n=%d)' % (kind, len(use)))
    lv = {a: st.fmean([r[kind + "_" + a] for r in use if kind + "_" + a in r]) for a in ARMS
          if any(kind + "_" + a in r for r in use)}
    print('  levels: ' + '  '.join('%s %+.2f' % (a, lv[a]) for a in ARMS if a in lv))
    if kind + "_rerank_hasgold" in use[0]:
        print('  rerank hits the gold passage: %.2f' % st.fmean(r.get(kind + "_rerank_hasgold", 0) for r in use))
    # key contrasts
    def c(a, b):
        m, p, n = paired([r.get(kind + "_" + a, None) is not None and r.get(kind + "_" + b, None) is not None
                          and (r[kind + "_" + a] - r[kind + "_" + b]) or None for r in use])
        return m, p, n
    for a, b in (("ansKV", "ansTX"), ("ansKV", "none"), ("rerank", "ansKV"), ("full", "ansKV"),
                 ("thinkKV", "ansKV"), ("thinkKV", "ansTX"), ("ansKV", "p_ansKV")):
        d = [r[kind + "_" + a] - r[kind + "_" + b] for r in use
             if kind + "_" + a in r and kind + "_" + b in r]
        if len(d) >= 2:
            m = st.fmean(d); sd = st.pstdev(d) or 1e-9; t = m / (sd / math.sqrt(len(d)))
            p = math.erfc(abs(t) / math.sqrt(2))
            print('  %-14s %+7.3f  p=%.4f  n=%d%s' % ('%s-%s' % (a, b), m, p, len(d), '  <<<' if p < .05 else ''))
    # recovery rate: fraction where the arm "recovers" the answer (log-odds > +1 = clearly prefers gold)
    rec = lambda a: st.fmean([1.0 if r.get(kind + "_" + a, -99) > 1 else 0.0 for r in use if kind + "_" + a in r])
    print('  recovery rate (log-odds>+1): full %.2f  ansKV %.2f  ansTX %.2f  rerank %.2f  none %.2f'
          % (rec('full'), rec('ansKV'), rec('ansTX'), rec('rerank'), rec('none')))
# negative source
print('\nneg_src: %s' % dict(collections.Counter(r.get('neg_src') for r in rows)))
