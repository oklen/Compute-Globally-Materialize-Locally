"""natres_extra.py -- robustness of the natural-occurrence read-out (paper App. I, Table e2nat).
Sign split / clusters / nV / category, the encode-time context ladder (real vs placebo),
and the split by UNSERVED value rows.  Reuses natres.py's parse + join (everything above the gate)."""
import json, sys, collections, statistics as st
sys.argv = ['x']; sys.path.insert(0, '.')
exec(open('natres.py').read().split('# ---- FREE GATE ----')[0])
lo = lambda r, k, h, a: r['c%d_%d_%s' % (k, h, a)]
def wcr(rows, d):
    cs = sorted({r['conv'] for r in rows}, key=int); ix = {c: i for i, c in enumerate(cs)}
    w = W(d, [ix[r['conv']] for r in rows], G=len(cs)); return w.mean, w.p()
for h in (0, 4, 16):
    use = [r for r in R if r['nV'] >= 1 and not r.get('v_in_hot_%d' % h) and 'c5_%d_iso' % h in r]
    d = [lo(r, 5, h, 'bhot') - lo(r, 5, h, 'self') for r in use]
    pos = sum(x > 0 for x in d); neg = sum(x < 0 for x in d); zero = sum(x == 0 for x in d)
    tr = sorted(d)[len(d) // 20:-len(d) // 20]
    byc = collections.defaultdict(list)
    for r, x in zip(use, d): byc[r['conv']].append(x)
    cm = [st.fmean(byc[c]) for c in sorted(byc, key=int)]
    print('\nhot=%d nV>=1 n=%d  bhot-self: mean %+.3f median %+.3f trimmed5%% %+.3f  sign +/0/- = %d/%d/%d  cluster means %s (%d/10 positive)'
          % (h, len(use), st.fmean(d), st.median(d), st.fmean(tr), pos, zero, neg, ' '.join('%+.2f' % c for c in cm), sum(c > 0 for c in cm)))
    for name, sel in (('nV=1', lambda r: r['nV'] == 1), ('nV>=2', lambda r: r['nV'] >= 2)):
        u = [r for r in use if sel(r)]
        if len(u) < 20: continue
        m, p = wcr(u, [lo(r, 5, h, 'bhot') - lo(r, 5, h, 'self') for r in u]); mp, pp = wcr(u, [lo(r, 5, h, 'p_bhot') - lo(r, 5, h, 'p_self') for r in u])
        print('   %-6s n=%3d  bhot-self %+.3f p=%.4f | placebo %+.3f p=%.4f' % (name, len(u), m, p, mp, pp))
    cats = collections.Counter(S[(r['conv'], r['qi'])].get('cat') for r in use)
    for c in sorted(cats, key=lambda x: -cats[x]):
        u = [r for r in use if S[(r['conv'], r['qi'])].get('cat') == c]
        if len(u) < 20: continue
        m, p = wcr(u, [lo(r, 5, h, 'bhot') - lo(r, 5, h, 'self') for r in u]); mp, pp = wcr(u, [lo(r, 5, h, 'p_bhot') - lo(r, 5, h, 'p_self') for r in u])
        print('   cat=%-4s n=%3d  bhot-self %+.3f p=%.4f | placebo %+.3f p=%.4f' % (c, len(u), m, p, mp, pp))
print('\n=== encode-time context ladder (log-odds levels), real vs placebo ===')
for h in (0, 4, 16):
    for name, sel in (('nV=0', lambda r: r['nV'] == 0), ('nV>=1', lambda r: r['nV'] >= 1)):
        use = [r for r in R if sel(r) and not r.get('v_in_hot_%d' % h) and 'c5_%d_iso' % h in r]
        L = lambda a: st.fmean([lo(r, 5, h, a) for r in use])
        print('hot=%2d %-5s n=%3d  real: iso %6.2f -> self %6.2f -> bhot %6.2f   placebo: p_iso %5.2f -> p_self %5.2f -> p_bhot %5.2f'
              % (h, name, len(use), L('iso'), L('self'), L('bhot'), L('p_iso'), L('p_self'), L('p_bhot')))
print('\n=== bhot-self by UNSERVED value rows (nV_used - nV_served), nV>=1 stratum ===')
for h in (0, 4, 16):
    use = [r for r in R if r['nV'] >= 1 and not r.get('v_in_hot_%d' % h) and 'c5_%d_iso' % h in r]
    for r in use: r['nVun'] = S[(r['conv'], r['qi'])]['nV_used'] - r['nV']
    for name, sel in (('unserved=0', lambda r: r['nVun'] == 0), ('unserved=1', lambda r: r['nVun'] == 1), ('unserved>=2', lambda r: r['nVun'] >= 2)):
        u = [r for r in use if sel(r)]
        if len(u) < 20: continue
        m, p = wcr(u, [lo(r, 5, h, 'bhot') - lo(r, 5, h, 'self') for r in u])
        print('   hot=%2d %-12s n=%3d  bhot-self %+.3f p=%.4f' % (h, name, len(u), m, p))
