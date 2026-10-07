"""chunkaudit.py -- did SPRAG_ENCODE_CHUNK drift across the CoT run, and did it matter?

Background.  `harv` gathers its KV from the ONE full-conversation prefill, which is computed in
chunks of SPRAG_ENCODE_CHUNK tokens.  Change the chunk size and that prefill changes slightly,
so `harv` changes -- and ONLY `harv`.  `iso` and `self` re-encode the served rows themselves and
never touch the chunked prefill.

The CoT run drifted: runners up to e2cot4_par.sh used 4096 (the value e2k_par.sh used to make
e2_k.json), and everything from e2cot4_rest.sh on used 2048.

TWO WAYS TO STRATIFY, AND ONLY ONE IS RIGHT.

  WRONG -- per item, by whether its no-CoT harv equals e2_k.json.  Log-odds are quantised to 1/8;
  on many items the chunk perturbation falls below that step and rounds away.  So "matches" means
  "the perturbation happened to vanish", not "ran at 4096".  That is selection on an
  outcome-adjacent variable, and it manufactured a scary +0.113 vs +0.000 split in `self - iso`
  -- a contrast that provably cannot depend on chunk at all.

  RIGHT -- by SOURCE LOG, which maps to a runner, which maps to a chunk.  cot_sh0.log is 0/83
  (a whole log at 4096); every other log is a mixture, i.e. it ran at 2048 and only some items
  moved.

Run from a directory holding cot_*.log, cot_all.json and e2_k.json.
"""
import re, json, glob, collections, sys, statistics as st
sys.path.insert(0, '.')
from wcrfast import W

K = {(str(r['conv']), int(r['qi'])): r for r in json.load(open('e2_k.json'))['rows']}

# ---- 1. control: iso/self must equal e2_k at ANY chunk; harv need not ----
NOCOT = re.compile(r'^\[e2k\] (\S+) q(\d+) \| k1 .*? k5 iso=(\S+) self=(\S+) harv=(\S+) ')
per = {}
for f in sorted(glob.glob('cot_*.log')):
    n = hb = sb = ib = 0
    for line in open(f, errors='ignore'):
        m = NOCOT.match(line)
        if not m:
            continue
        ref = K.get((m.group(1), int(m.group(2))))
        if not ref:
            continue
        n += 1
        ib += abs(float(m.group(3)) - ref['k5_iso_lo']) > 0
        sb += abs(float(m.group(4)) - ref['k5_self_lo']) > 0
        hb += abs(float(m.group(5)) - ref['k5_harv_lo']) > 0
    if n:
        per[f] = (n, ib, sb, hb)
print('%-38s %6s %7s %8s %8s  %s' % ('log', 'items', 'iso≠', 'self≠', 'harv≠', 'verdict'))
tot = [0, 0, 0, 0]
for f, (n, ib, sb, hb) in per.items():
    for i, v in enumerate((n, ib, sb, hb)):
        tot[i] += v
    print('%-38s %6d %7d %8d %8d  %s'
          % (f, n, ib, sb, hb, '4096' if hb == 0 else '2048 (%d/%d harv moved)' % (hb, n)))
print('%-38s %6d %7d %8d %8d' % ('total', *tot))
print('=> iso/self mismatches against e2_k: %d/%d -- "chunking only moves harv" holds in the data.'
      % (tot[1] + tot[2], 2 * tot[0]))

# ---- 2. stratified contrasts, by source log ----
KEY = re.compile(r'^\[e2cot\] (\S+) q(\d+) \|')
src = {}
for f in sorted(glob.glob('cot_*.log')):
    for line in open(f, errors='ignore'):
        m = KEY.match(line)
        if m:
            src[(m.group(1), int(m.group(2)))] = f
R = json.load(open('cot_all.json'))
for r in R:
    f = src.get((r['conv'], r['qi']), '?')
    r['chunk'] = '4096' if f == 'cot_sh0.log' else '2048'

acc = lambda r, pre, a: 1.0 if r[pre + a] > 0 else 0.0
def wcr(rows, d):
    cs = sorted({r['conv'] for r in rows}); ix = {c: i for i, c in enumerate(cs)}
    w = W(d, [ix[r['conv']] for r in rows], G=len(cs)); return w.mean, w.p(), len(cs)
def con(rows, A, B, pre='c_'): return [acc(r, pre, A) - acc(r, pre, B) for r in rows]

print('\n%-24s %5s %4s %16s %16s %18s'
      % ('stratum', 'n', 'G', 'self-iso', 'harv-self', 'paired harv-self'))
for name, rows in (('all', R),
                   ('chunk=4096 (shard 0)', [r for r in R if r['chunk'] == '4096']),
                   ('chunk=2048 (shard 1,2)', [r for r in R if r['chunk'] == '2048'])):
    a, pa, G = wcr(rows, con(rows, 'self', 'iso'))
    b, pb, _ = wcr(rows, con(rows, 'harv', 'self'))
    d = [(acc(r, 'c_', 'harv') - acc(r, 'c_', 'self'))
         - (acc(r, 'n_', 'harv') - acc(r, 'n_', 'self')) for r in rows]
    c, pc, _ = wcr(rows, d)
    print('%-24s %5d %4d %+9.4f p=%.3f %+9.4f p=%.3f %+10.4f p=%.3f'
          % (name, len(rows), G, a, pa, b, pb, c, pc))
print('Note: the two strata share no cluster (4096 = shard 0: 0/3/6/9; 2048 = shards 1,2: 1/2/4/5/7/8),')
print('    so chunk is collinear with shard and is not a clean contrast; at G=4 the smallest WCR p is 0.125 and the interval is unbounded, so no CI is reported.')
