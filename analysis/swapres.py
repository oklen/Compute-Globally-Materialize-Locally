"""swapres.py -- read-out of the bridge experiment (kv_e2swap.py logs).

    o_*  original world     c_*  counterfactual world (gold -> neg everywhere, R3)
    lo = log p(gold) - log p(neg); in the cf world the evidence says neg, so lo should be < 0.

    Delta_KV   = o_bhot - c_bhot        Delta_text = o_self - c_self
    I          = Delta_KV - Delta_text  = (bhot - self)_orig - (bhot - self)_cf
    evidence-driven: (bhot-self)_cf mirrors (~ -1.87) => I ~ +3.7 ; prior-driven: I ~ 0
"""
import re, glob, sys, statistics as st, collections
sys.path.insert(0, '.')
from wcrfast import W

ARMS = ("bhot", "self", "harv")
HEAD = re.compile(r'^\[e2swap\] (\S+) q(\d+) \| (.*?) \| vhot=(\d+) hchg=(\d+) chg=(\d+) pchg=(\d) drift=(-?\d+) ')
CELL = re.compile(r'k(\d+)h(\d+) ((?:\S+=\S+ ?)+)')

rows = {}
for f in sorted(glob.glob('sw_sh*.log')):
    for line in open(f, errors='ignore'):
        m = HEAD.match(line)
        if not m:
            continue
        r = {'conv': m.group(1), 'qi': int(m.group(2)), 'chg': int(m.group(6)), 'pchg': int(m.group(7)),
             'drift': int(m.group(8))}
        cells = CELL.findall(m.group(3))
        hots = sorted({int(c[1]) for c in cells})
        for c in cells:
            for kv in c[2].split():
                k, v = kv.split('=')
                r['c%s_%s_%s' % (c[0], c[1], k)] = float(v)
        for i, h in enumerate(hots):
            r['v_in_hot_%d' % h] = int(m.group(4)[i]); r['hot_chg_%d' % h] = int(m.group(5)[i])
        rows[(r['conv'], r['qi'])] = r
R = sorted(rows.values(), key=lambda r: (int(r['conv']), r['qi']))
print('n=%d  clusters=%d  median rewritten rows=%s  max=%d  placebo rows rewritten=%d  median token drift=%s'
      % (len(R), len({r['conv'] for r in R}), st.median([r['chg'] for r in R]) if R else '-',
         max(r['chg'] for r in R) if R else 0, sum(r['pchg'] for r in R), st.median([r['drift'] for r in R]) if R else '-'))

def wcr(rows, d, alpha=.05):
    cs = sorted({r['conv'] for r in rows}, key=int); ix = {c: i for i, c in enumerate(cs)}
    G = len(cs); w = W(d, [ix[r['conv']] for r in rows], G=G)
    if 2 / 2 ** G > alpha:
        return w.mean, w.p(), None
    a, b = w.ci(alpha=alpha)
    return w.mean, w.p(), ((a, b) if b - a < 1e6 else None)
def fci(c): return '[%+.3f,%+.3f]' % c if c else '[G too small]'
def rep(name, rows, d):
    m, p, c = wcr(rows, d)
    print('  %-34s %+8.3f  p=%.4f  %s%s' % (name, m, p, fci(c), '  <<<' if p < .05 else ''))

for k, h in ((5, 0), (5, 4), (5, 16)):
    use = [r for r in R if not r.get('v_in_hot_%d' % h) and 'c%d_%d_o_bhot' % (k, h) in r]
    if len(use) < 20:
        continue
    g = lambda r, w, a: r['c%d_%d_%s%s' % (k, h, w, a)]
    print('\n' + '=' * 78 + '\nk=%d hot=%d  n=%d  clusters=%d' % (k, h, len(use), len({r['conv'] for r in use})))
    L = lambda w, a: st.fmean([g(r, w, a) for r in use])
    print('  levels  orig: bhot %+6.2f self %+6.2f harv %+6.2f   cf: bhot %+6.2f self %+6.2f harv %+6.2f'
          % (L('o_', 'bhot'), L('o_', 'self'), L('o_', 'harv'), L('c_', 'bhot'), L('c_', 'self'), L('c_', 'harv')))
    print('  positive control  cf world self<0: %.3f   bhot<0: %.3f   (after the swap the evidence says neg, the readout should turn negative)'
          % (st.fmean([g(r, 'c_', 'self') < 0 for r in use]), st.fmean([g(r, 'c_', 'bhot') < 0 for r in use])))
    rep('(bhot-self)_orig', use, [g(r, 'o_', 'bhot') - g(r, 'o_', 'self') for r in use])
    rep('(bhot-self)_cf', use, [g(r, 'c_', 'bhot') - g(r, 'c_', 'self') for r in use])
    rep('Delta_KV = o_bhot - c_bhot', use, [g(r, 'o_', 'bhot') - g(r, 'c_', 'bhot') for r in use])
    rep('Delta_text = o_self - c_self', use, [g(r, 'o_', 'self') - g(r, 'c_', 'self') for r in use])
    rep('I = Delta_KV - Delta_text', use, [(g(r, 'o_', 'bhot') - g(r, 'o_', 'self')) - (g(r, 'c_', 'bhot') - g(r, 'c_', 'self')) for r in use])
    if h > 0:
        rep('(harv-bhot)_orig', use, [g(r, 'o_', 'harv') - g(r, 'o_', 'bhot') for r in use])
        rep('(harv-bhot)_cf', use, [g(r, 'c_', 'harv') - g(r, 'c_', 'bhot') for r in use])
    pu = [r for r in use if not r['pchg'] and 'c%d_%d_o_p_bhot' % (k, h) in r]
    if len(pu) >= 20:
        print('  placebo (%d items whose placebo rows were not rewritten)' % len(pu))
        rep('  (p_bhot-p_self)_orig', pu, [g(r, 'o_', 'p_bhot') - g(r, 'o_', 'p_self') for r in pu])
        rep('  (p_bhot-p_self)_cf', pu, [g(r, 'c_', 'p_bhot') - g(r, 'c_', 'p_self') for r in pu])
        rep('  I_placebo', pu, [(g(r, 'o_', 'p_bhot') - g(r, 'o_', 'p_self')) - (g(r, 'c_', 'p_bhot') - g(r, 'c_', 'p_self')) for r in pu])
    # split by whether the model resisted (cf self still > 0)
    res = [r for r in use if g(r, 'c_', 'self') > 0]
    if len(res) >= 20:
        print('  resisting layer (cf world, self still prefers gold, n=%d):' % len(res))
        rep('  I (resisting layer)', res, [(g(r, 'o_', 'bhot') - g(r, 'o_', 'self')) - (g(r, 'c_', 'bhot') - g(r, 'c_', 'self')) for r in res])
    fol = [r for r in use if g(r, 'c_', 'self') < 0]
    if len(fol) >= 20:
        rep('  I (following layer, cf self<0)', fol, [(g(r, 'o_', 'bhot') - g(r, 'o_', 'self')) - (g(r, 'c_', 'bhot') - g(r, 'c_', 'self')) for r in fol])
