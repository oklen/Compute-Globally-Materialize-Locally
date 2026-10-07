"""hotres2.py -- read-out for the deployed b_hot arm.

Two contrasts this run exists for:
    harv - bhot   what HARVESTING the hot tail costs versus RECOMPUTING it
    bhot - self   the materialization channel AT THE DEPLOYED OPERATING POINT

`self` is the right control for the second: identical served text, identical positions, identical
prefix -- the only difference is that `self` re-encodes everything while `bhot` takes the non-hot
rows from the full-conversation prefill.  So `bhot - self` is exactly "what did the rest of the
conversation deposit in those rows", measured at the operating point production actually uses.
"""
import json, sys, statistics as st, collections
sys.path.insert(0, '.')
from wcrfast import W

R = json.load(open('hot2_all.json'))
CONV = sorted({r['conv'] for r in R}, key=int)
CELLS = sorted({(int(k[1:].split('_')[0]), int(k[1:].split('_')[1]))
                for r in R for k in r if k.startswith('c') and k.endswith('_bhot')})
HS = sorted({h for k, h in CELLS if k == 5})

def wcr(rows, d, alpha=0.05):
    cs = sorted({r['conv'] for r in rows}, key=int); ix = {c: i for i, c in enumerate(cs)}
    G = len(cs)
    w = W(d, [ix[r['conv']] for r in rows], G=G)
    if 2 / 2 ** G > alpha:
        return w.mean, w.p(), None
    lo, hi = w.ci(alpha=alpha)
    return (w.mean, w.p(), (lo, hi) if hi - lo < 1e6 else None)

def fci(ci): return '[%+.4f,%+.4f]' % ci if ci else '[G too small, no interval]'
def lo(r, k, h, a): return r['c%d_%d_%s' % (k, h, a)]
def ac(r, k, h, a): return 1.0 if r['c%d_%d_%s' % (k, h, a)] > 0 else 0.0

print('n=%d  clusters=%d  cells=%s' % (len(R), len(CONV), CELLS))

print('\n########## instrument: absolute levels of the four arms (before any contrast) ##########')
print('  %-12s %8s %8s %8s %8s   %s' % ('cell', 'iso', 'self', 'harv', 'bhot', 'hot-window lift (vs hot=0)'))
base = {}
for k, h in CELLS:
    use = [r for r in R if 'c%d_%d_iso' % (k, h) in r]
    v = {a: st.fmean([r['c%d_%d_%s' % (k, h, a)] for r in use])
         for a in ('iso', 'self', 'harv', 'bhot')}
    if h == 0:
        base = dict(v)
    d = '  '.join('%s %+.2f' % (a, v[a] - base[a]) for a in ('iso', 'self', 'harv', 'bhot')) if h else ''
    print('  k=%d hot=%-4d %8.3f %8.3f %8.3f %8.3f   %s'
          % (k, h, v['iso'], v['self'], v['harv'], v['bhot'], d))

PAIRS = (('bhot', 'self'), ('harv', 'bhot'), ('harv', 'self'), ('self', 'iso'), ('bhot', 'iso'),
         ('p_bhot', 'p_self'), ('p_harv', 'p_bhot'), ('p_bhot', 'p_iso'))
for ruler, f in (('log-odds', lo), ('accuracy', ac)):
    print('\n########## %s ##########' % ruler)
    for k, h in CELLS:
        use = [r for r in R if not r.get('v_in_hot_%d' % h) and 'c%d_%d_bhot' % (k, h) in r]
        print('\n=== k=%d hot=%-2d  n=%d (dropped %d with V in the hot window) ===' % (k, h, len(use), len(R) - len(use)))
        for A, B in PAIRS:
            m, p, ci = wcr(use, [f(r, k, h, A) - f(r, k, h, B) for r in use])
            print('  %-14s %+8.4f  p=%.4f  %s%s'
                  % ('%s-%s' % (A, B), m, p, fci(ci), ' <<<' if p < .05 else ''))
    common = [r for r in R if all(not r.get('v_in_hot_%d' % h) and 'c5_%d_bhot' % h in r for h in HS)]
    if len(HS) > 1 and common:
        print('\n=== dose response (k=5, common paired sample n=%d) ===' % len(common))
        for A, B in (('bhot', 'self'), ('harv', 'bhot'), ('harv', 'self'), ('self', 'iso'),
                     ('p_bhot', 'p_self'), ('p_harv', 'p_bhot')):
            out = []
            for h in HS:
                m, p, _ = wcr(common, [f(r, 5, h, A) - f(r, 5, h, B) for r in common])
                out.append('hot=%-2d %+7.4f (p=%.3f)' % (h, m, p))
            print('  %-14s %s' % ('%s-%s' % (A, B), '  |  '.join(out)))
        print('\n  --- paired per item: does the hot window change this contrast (relative to hot=0) ---')
        for A, B in (('bhot', 'self'), ('harv', 'bhot'), ('harv', 'self')):
            for h in HS[1:]:
                d = [(f(r, 5, h, A) - f(r, 5, h, B)) - (f(r, 5, 0, A) - f(r, 5, 0, B)) for r in common]
                m, p, ci = wcr(common, d)
                print('  %-12s hot=%-2d − hot=0: %+8.4f  p=%.4f  %s%s'
                      % ('%s-%s' % (A, B), h, m, p, fci(ci), ' <<<' if p < .05 else ''))
