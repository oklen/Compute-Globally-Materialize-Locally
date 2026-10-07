"""natres.py -- the natural-occurrence read-out.

The forced condition (kv_e2sets) removed every value row before ranking, so `harv == self`
there is a parity statement about a task sitting near the floor.  The natural condition
ranks the full pool and lets the retriever hit or miss, which splits the SAME 1269 questions into

    nV_served == 0   the retriever naturally reached no value row  (511 items, 40.3%)
    nV_served >= 1   the answer IS in the served rows              (758 items, 59.7%)

The second stratum is the point: accuracy there is far off the floor, so parity measured in it is
parity at an operating point that works.

FREE GATE.  On the 511 items whose natural serve set is IDENTICAL to the forced one, every number
must reproduce the forced run bit-for-bit -- same rows, same positions, same everything.  If it
does not, the two runs are not comparable and no stratum contrast means anything.
"""
import re, json, glob, sys, collections, statistics as st
sys.path.insert(0, '.')
from wcrfast import W

ARMS = ("iso", "self", "harv", "bhot", "p_iso", "p_self", "p_harv", "p_bhot")
HEAD = re.compile(r'^\[e2hot2\] (\S+) q(\d+) \| (.*?) \| vhot=(\d+) nu=(-?\d+) gold=')
CELL = re.compile(r'k(\d+)h(\d+) ' + ' '.join(r'%s=(\S+)' % a for a in ARMS))

def parse(patterns):
    rows, hots = {}, []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f, errors='ignore'):
                m = HEAD.match(line)
                if not m:
                    continue
                cells = CELL.findall(m.group(3))
                if not cells:
                    continue
                r = {'conv': m.group(1), 'qi': int(m.group(2))}
                for c in cells:
                    for a, v in zip(ARMS, c[2:]):
                        r['c%s_%s_%s' % (c[0], c[1], a)] = float(v)
                if not hots:
                    hots = sorted({int(c[1]) for c in cells})
                for i, h in enumerate(hots):
                    if i < len(m.group(4)):
                        r['v_in_hot_%d' % h] = int(m.group(4)[i])
                rows[(r['conv'], r['qi'])] = r
    return rows, hots

NAT, HOTS = parse(['nat_*.log'])
FRC, _ = parse(['hot2_*.log'])
S = {(str(i['conv']), i['qi']): i for i in json.load(open('e2_natsets.json'))['items']}
F = {(str(i['conv']), i['qi']): i for i in json.load(open('e2_sets.json'))['items']}

R = []
for k, r in sorted(NAT.items(), key=lambda x: (int(x[0][0]), x[0][1])):
    it = S.get(k)
    if not it:
        continue
    r['nV'] = it['nV_served']; r['pnV'] = it['p_nV_served']
    r['same_as_forced'] = int(k in F and set(it['serve']) == set(F[k]['serve']))
    R.append(r)
print('n=%d  clusters=%d' % (len(R), len({r['conv'] for r in R})))
print('nV_served: %s' % dict(sorted(collections.Counter(r['nV'] for r in R).items())))
print('serve set identical to the forced set: %d (%.1f%%)'
      % (sum(r['same_as_forced'] for r in R), 100 * sum(r['same_as_forced'] for r in R) / len(R)))

# ---- FREE GATE ----
# Real arms: identical serve set  =>  identical numbers.  Placebo arms: the natural placebo is
# rebuilt from the natural pool, so they are only comparable where the placebo set is ALSO identical.
CELLF = re.compile(r'^c\d+_\d+_(p_)?(iso|self|harv|bhot)$')
n = bad = np_ = badp = 0
for r in R:
    if not r['same_as_forced']:
        continue
    o = FRC.get((r['conv'], r['qi']))
    if not o:
        continue
    k = (r['conv'], r['qi'])
    same_pl = set(S[k]['placebo']) == set(F[k]['placebo'])
    for f in r:
        m = CELLF.match(f)
        if not m or f not in o:
            continue
        if m.group(1):
            if same_pl:
                np_ += 1; badp += (r[f] != o[f])
        else:
            n += 1; bad += (r[f] != o[f])
print('gate  items with the same serve set vs the forced run: real arms %d numbers, %d mismatches; placebo arms on items whose placebo set is also the same %d numbers, %d mismatches  (both must be 0)'
      % (n, bad, np_, badp))
if (n and bad) or (np_ and badp):
    print('  => the two runs are not comparable; all stratified contrasts below are void'); sys.exit(1)

acc = lambda r, k, h, a: 1.0 if r['c%d_%d_%s' % (k, h, a)] > 0 else 0.0
lo = lambda r, k, h, a: r['c%d_%d_%s' % (k, h, a)]
def wcr(rows, d, alpha=.05):
    cs = sorted({r['conv'] for r in rows}, key=int); ix = {c: i for i, c in enumerate(cs)}
    G = len(cs); w = W(d, [ix[r['conv']] for r in rows], G=G)
    if 2 / 2 ** G > alpha:
        return w.mean, w.p(), None
    a, b = w.ci(alpha=alpha)
    return w.mean, w.p(), ((a, b) if b - a < 1e6 else None)
def fci(c): return '[%+.4f,%+.4f]' % c if c else '[G too small]'

STRATA = (('naturally zero V (nV=0)', lambda r: r['nV'] == 0),
          ('some V served (nV>=1)', lambda r: r['nV'] >= 1),
          ('all', lambda r: True))
for k, h in ((5, 0), (5, 4), (5, 16)):
    print('\n' + '=' * 78)
    print('k=%d hot=%d' % (k, h))
    for name, sel in STRATA:
        use = [r for r in R if sel(r) and not r.get('v_in_hot_%d' % h) and 'c%d_%d_iso' % (k, h) in r]
        if len(use) < 20:
            continue
        lvl = {a: st.fmean([acc(r, k, h, a) for r in use]) for a in ('iso', 'self', 'harv', 'bhot')}
        print('\n--- %s  n=%d  clusters=%d ---' % (name, len(use), len({r['conv'] for r in use})))
        print('  accuracy levels: iso %.3f  self %.3f  harv %.3f  bhot %.3f'
              % (lvl['iso'], lvl['self'], lvl['harv'], lvl['bhot']))
        for A, B in (('self', 'iso'), ('harv', 'self'), ('bhot', 'self'), ('harv', 'bhot'),
                     ('p_harv', 'p_self'), ('p_bhot', 'p_self')):
            ma, pa, ca = wcr(use, [acc(r, k, h, A) - acc(r, k, h, B) for r in use])
            ml, pl, cl = wcr(use, [lo(r, k, h, A) - lo(r, k, h, B) for r in use])
            print('  %-14s accuracy %+7.4f p=%.4f %-22s | log-odds %+8.4f p=%.4f%s'
                  % ('%s-%s' % (A, B), ma, pa, fci(ca), ml, pl, ' <<<' if min(pa, pl) < .05 else ''))
