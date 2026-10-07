"""citeres.py -- read-out of the counterfactual citation experiment (kv_cite.py logs).

Per (sentence, cited passage) the key quantities (forced-choice log-odds of gold vs neg):
    l_none  no-context prior      (high => parametric memory)
    dA      = lo(all) - lo(d_A value swapped)     dependence on the answer passage
    dB      = lo(+d_C) - lo(+d_C, bridge in d_B swapped)   bridge load-bearing?
    dP      placebo (a distractor entity swapped)  must be ~0
    dup a/b/both   double-source: which copy did the model use?
    plant_d planted positive control (model can't know v*)  must be large

Faithfulness = the citation is load-bearing.  We classify each item and cross-tab with whether the
model actually CITED d_A.  Bounded readout, so a large dA is real dependence, not a length artifact.
"""
import json, glob, sys, statistics as st, collections
sys.path.insert(0, '.')
try:
    from wcrfast import W
except Exception:
    W = None

rows = []
for f in sorted(glob.glob('cite_full*.log')) or sorted(glob.glob('cite*.log')):
    for line in open(f, errors='ignore'):
        pass
# prefer the JSON if present (has neg_src, plant, all fields)
import re as _re
rows = []
seen = set()
# 1) the JSON (continuation run = full field set)
for cand in ('cite.json',):
    for f in glob.glob(cand):
        for r in json.load(open(f))['rows']:
            if r.get('status') == 'ok':
                rows.append(r); seen.add(r['id'])
# 2) part-1 rows survive only in the log (dir was renamed then the JSON overwritten): parse the line
LINE = _re.compile(r'^\[cite\] (\S+) \| citedA=(\d+) citedB=(\d+) n_cited=(\d+) \| '
                   r'l_none=([-+0-9.]+) l0=([-+0-9.]+) dA=([-+0-9.]+) dB=([-+0-9.]+) dP=(\S+) \| '
                   r'dup a/b/both=(\S+)/(\S+)/(\S+) \| plant_d=(\S+)')
def _f(x):
    return None if x == 'na' else float(x)
for f in glob.glob('cite_p1_full.log') + glob.glob('cite_p1*.log'):
    for line in open(f, errors='ignore'):
        m = LINE.match(line)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        rows.append(dict(status='ok', id=m.group(1), citedA=int(m.group(2)), citedB=int(m.group(3)),
                         n_cited=int(m.group(4)), l_none=float(m.group(5)), l0=float(m.group(6)),
                         dA=float(m.group(7)), dB=float(m.group(8)), dP=_f(m.group(9)),
                         d6a=_f(m.group(10)), d6b=_f(m.group(11)), d6c=_f(m.group(12)),
                         plant_d=_f(m.group(13)), neg_src='p1_log'))
if not rows:
    print('no cite rows found'); sys.exit(1)
J = {'skip': 'merged p1+cont'}
print('n_ok=%d  skip=%s  neg_src: %s' % (len(rows), J.get('skip'),
      dict(collections.Counter(r.get('neg_src') for r in rows))))

# thresholds (log-odds): the forced choice saturates near +-27, so use fractions of the item's own
# no-swap preference l0 rather than absolute cuts.
def cls(r):
    dA, ln, l0 = r['dA'], r['l_none'], r['l0']
    load = dA >= 3.0                       # swapping d_A's value moved the answer by >=3 log-odds
    param = ln >= 3.0                      # the model already preferred gold without any passage
    if load:
        return 'load_bearing'
    if param:
        return 'parametric'                # cited but the model knew it anyway
    return 'weak'                          # neither strongly used nor clearly parametric

for name, sel in (('all', lambda r: True),
                  ('cited d_A', lambda r: r.get('citedA')),
                  ('did NOT cite d_A', lambda r: not r.get('citedA'))):
    use = [r for r in rows if sel(r)]
    if not use:
        continue
    C = collections.Counter(cls(r) for r in use)
    m = lambda k: st.fmean([r[k] for r in use if r.get(k) is not None])
    print('\n=== %s  (n=%d) ===' % (name, len(use)))
    print('  class: %s' % dict(C))
    print('  citedA rate %.3f  citedB rate %.3f  mean n_cited %.2f'
          % (st.fmean(r['citedA'] for r in use), st.fmean(r['citedB'] for r in use), m('n_cited')))
    print('  dA %+.2f  dB %+.2f  dP %+.2f  l_none %+.2f' % (m('dA'), m('dB'), m('dP'), m('l_none')))
    print('  double-source  swapA %+.2f  swapE %+.2f  both %+.2f' % (m('d6a'), m('d6b'), m('d6c')))

# the headline contingency: load-bearing vs cited
print('\n=== faithfulness contingency (load-bearing dA>=3  x  cited d_A) ===')
tab = collections.Counter((cls(r) == 'load_bearing', bool(r.get('citedA'))) for r in rows)
print('  load-bearing & cited     : %d' % tab[(True, True)])
print('  load-bearing & NOT cited : %d   <- MISSED CITATIONS (model used it, did not cite)' % tab[(True, False)])
print('  parametric/weak & cited  : %d   <- DECORATIVE (cited, not load-bearing)'
      % sum(1 for r in rows if cls(r) != 'load_bearing' and r.get('citedA')))
print('  parametric/weak & not    : %d' % sum(1 for r in rows if cls(r) != 'load_bearing' and not r.get('citedA')))
cited = [r for r in rows if r.get('citedA')]
if cited:
    dec = sum(1 for r in cited if cls(r) != 'load_bearing')
    print('  => of citations to d_A, %.1f%% are NOT load-bearing (decorative/post-rationalized)'
          % (100 * dec / len(cited)))
# planted positive control
pl = [r for r in rows if r.get('plant_d') is not None]
if pl:
    print('\n=== planted positive control (model cannot know v*) ===')
    print('  n=%d  mean plant_d %+.2f  frac >=3: %.2f  plant_citedA rate %.2f'
          % (len(pl), st.fmean(r['plant_d'] for r in pl),
             st.fmean(1.0 if r['plant_d'] >= 3 else 0.0 for r in pl),
             st.fmean(r.get('plant_citedA', 0) for r in pl)))
# placebo distribution
print('\n=== placebo dP (must be ~0) ===')
dP = [r['dP'] for r in rows if r.get('dP') is not None]
print('  mean %+.3f  median %+.3f  frac |dP|>=3: %.3f' % (st.fmean(dP), st.median(dP),
      st.fmean(1.0 if abs(x) >= 3 else 0.0 for x in dP)))
# bridge
print('\n=== bridge dB (is the un-cited bridge passage load-bearing?) ===')
dB = [r['dB'] for r in rows if r.get('dB') is not None]
print('  mean %+.3f  frac >=3: %.3f  (of items where d_B not cited: %d)'
      % (st.fmean(dB), st.fmean(1.0 if x >= 3 else 0.0 for x in dB),
         sum(1 for r in rows if not r.get('citedB'))))
