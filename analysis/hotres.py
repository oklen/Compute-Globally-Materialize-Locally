"""hotres.py -- the read-out for E2 at hot>0.

Reports, per (k, hot) cell and on BOTH rulers:
    self - iso    the served rows seeing each other        (not materialization)
    harv - self   everything else in the conversation      (THE materialization channel)
    harv - iso    their sum, the published PRIMARY
plus the two placebo contrasts, WCR with cluster = conversation.

The dose-response on `harv - self` across hot = 0 / 4 / 16 is the point of the run: one flat
point cannot separate "no effect" from "hot window too small", three can.  Every cell is
restricted to items whose V set does not intersect that cell's hot window, and the dose-response
is additionally restricted to the COMMON sample so the three points are paired.
"""
import json, sys, statistics as st, collections
sys.path.insert(0, '.')
from wcrfast import W

R = json.load(open('hot_all.json'))
CONV = sorted({r['conv'] for r in R}, key=int)
IDX = {c: i for i, c in enumerate(CONV)}
# derive cells from key names c<k>_<h>_<arm>
CELLS = sorted({(int(k[1:].split('_')[0]), int(k[1:].split('_')[1]))
                for r in R for k in r if k.startswith('c') and k.endswith('_iso')})

def wcr(rows, d, alpha=0.05):
    """WCR mean, p, CI -- with the CI SUPPRESSED when it cannot exist.

    WCR enumerates all 2^G sign vectors, so the smallest attainable p is 2/2^G.  At G<=5 that is
    >= 0.0625 > alpha, i.e. the null is never rejected at any mu and the CI-inversion search runs
    away to +-1e9.  Printing that as an interval is worse than printing nothing -- it looks like
    a measured bound.  Return None and let the caller show a dash.
    """
    cs = sorted({r['conv'] for r in rows}, key=int); ix = {c: i for i, c in enumerate(cs)}
    G = len(cs)
    w = W(d, [ix[r['conv']] for r in rows], G=G)
    if 2 / 2 ** G > alpha:
        return w.mean, w.p(), None
    lo, hi = w.ci(alpha=alpha)
    span = hi - lo
    if not (span < 1e6):          # inversion failed to bracket -> not a bound
        return w.mean, w.p(), None
    return w.mean, w.p(), (lo, hi)


def fci(ci):
    return '[%+.4f,%+.4f]' % ci if ci else '[G too small, no WCR interval]' 

def lo(r, k, h, a): return r['c%d_%d_%s' % (k, h, a)]
def ac(r, k, h, a): return 1.0 if r['c%d_%d_%s' % (k, h, a)] > 0 else 0.0

print('n=%d  clusters=%d  cells=%s' % (len(R), len(CONV), CELLS))

# ---------------------------------------------------------------------------
# INSTRUMENT: did the hot window change the task at all?
#
# If every arm's absolute level is identical at hot=0 and hot=16, then a flat `harv - self`
# says nothing -- the hot window would simply not have been part of the measurement.  The
# absolute levels are the only thing in this run that can tell "no materialization channel"
# apart from "the manipulation never landed".  Report them BEFORE any contrast.
#
# Also report the real/placebo overlap.  The hot window is shared by construction (production
# serves the same recent turns whatever the query), so as hot grows the placebo serve set
# converges on the real one -- measured Jaccard 0.08 / 0.35 / 0.66 at hot = 0 / 4 / 16.  The
# placebo therefore DISCRIMINATES LESS at larger hot; that is a property of the design, not a
# result, and `harv - self` (whose control is `self`, same text same positions) is unaffected.
# ---------------------------------------------------------------------------
print('\n########## instrument: does the hot window change the task itself? ##########')
print('  %-10s %8s %8s %8s %8s %8s' % ('cell', 'iso lo', 'self lo', 'harv lo', 'mean|lo|', 'sat%'))
for k, h in CELLS:
    use = [r for r in R if 'c%d_%d_iso' % (k, h) in r]
    if not use:
        continue
    lv = [st.fmean([r['c%d_%d_%s' % (k, h, a)] for r in use]) for a in ('iso', 'self', 'harv')]
    mag = st.fmean([abs(r['c%d_%d_%s' % (k, h, a)]) for r in use for a in ('iso', 'self', 'harv')])
    sat = 100 * st.fmean([abs(r['c%d_%d_%s' % (k, h, a)]) >= 15
                          for r in use for a in ('iso', 'self', 'harv')])
    print('  k=%d hot=%-4d %8.3f %8.3f %8.3f %8.3f %7.1f%%' % (k, h, lv[0], lv[1], lv[2], mag, sat))
print('  => if the absolute arm levels do not differ between hot=0 and hot=16, the hot window never entered the measurement,'
      ' and no "harv-self is still zero" can count as evidence.')
for ruler, f in (('log-odds', lo), ('accuracy', ac)):
    print('\n########## %s ##########' % ruler)
    for k, h in CELLS:
        use = [r for r in R if not r.get('v_in_hot_%d' % h) and 'c%d_%d_iso' % (k, h) in r]
        if not use:
            continue
        print('\n=== k=%d hot=%-2d  n=%d (dropped %d with V in the hot window) ==='
              % (k, h, len(use), len(R) - len(use)))
        for A, B in (('self', 'iso'), ('harv', 'self'), ('harv', 'iso'),
                     ('p_self', 'p_iso'), ('p_harv', 'p_self'), ('p_harv', 'p_iso')):
            d = [f(r, k, h, A) - f(r, k, h, B) for r in use]
            m, p, ci = wcr(use, d)
            star = ' <<<' if p < .05 else ''
            print('  %-14s %+8.4f  p=%.4f  %s%s' % ('%s-%s' % (A, B), m, p, fci(ci), star))

    # ---- dose-response on the common paired sample ----
    hs = sorted({h for kk, h in CELLS if kk == 5})
    common = [r for r in R
              if all(not r.get('v_in_hot_%d' % h) and 'c5_%d_iso' % h in r for h in hs)]
    if len(hs) > 1 and common:
        print('\n=== dose response (k=5, common paired sample n=%d) ===' % len(common))
        for A, B in (('harv', 'self'), ('self', 'iso'), ('harv', 'iso'), ('p_harv', 'p_iso')):
            cells = []
            for h in hs:
                d = [f(r, 5, h, A) - f(r, 5, h, B) for r in common]
                m, p, _ = wcr(common, d)
                cells.append('hot=%-2d %+7.4f (p=%.3f)' % (h, m, p))
            print('  %-14s %s' % ('%s-%s' % (A, B), '  |  '.join(cells)))
        print('\n  --- paired per item: does the hot window change this contrast (relative to hot=0) ---')
        for A, B in (('harv', 'self'), ('self', 'iso'), ('harv', 'iso')):
            for h in hs[1:]:
                d = [(f(r, 5, h, A) - f(r, 5, h, B)) - (f(r, 5, 0, A) - f(r, 5, 0, B))
                     for r in common]
                m, p, ci = wcr(common, d)
                star = ' <<<' if p < .05 else ''
                print('  %-12s hot=%-2d − hot=0: %+8.4f  p=%.4f  %s%s'
                      % ('%s-%s' % (A, B), h, m, p, fci(ci), star))
