"""hotparse2.py -- rebuild the kv_e2hot2 table from LOGS (four arms + four placebos).

Same reason as hotparse/cotparse2: logs are the only copy that survives a worker reclaim.
Also runs the two identity self-checks that must hold before any contrast is read:
  * bhot == harv at hot=0   (no hot tail to recompute -> one computation)
  * iso/self/harv at every cell == what kv_e2hot measured   (this module only ADDS an arm)
"""
import re, json, glob, sys, collections, statistics as st

ARMS = ("iso", "self", "harv", "bhot", "p_iso", "p_self", "p_harv", "p_bhot")
HEAD = re.compile(r'^\[e2hot2\] (\S+) q(\d+) \| (.*?) \| vhot=(\d+) nu=(-?\d+) gold=')
CELL = re.compile(r'k(\d+)h(\d+) ' + ' '.join(r'%s=(\S+)' % a for a in ARMS))

rows, hots = {}, []
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'hot2_*.log'))
for f in files:
    for line in open(f, errors='ignore'):
        m = HEAD.match(line)
        if not m:
            continue
        cells = CELL.findall(m.group(3))
        if not cells:
            continue
        r = {'conv': m.group(1), 'qi': int(m.group(2)), 'nunit': int(m.group(5))}
        for c in cells:
            k, h = int(c[0]), int(c[1])
            for a, v in zip(ARMS, c[2:]):
                r['c%d_%d_%s' % (k, h, a)] = float(v)
        if not hots:
            hots = sorted({int(c[1]) for c in cells})
        for i, h in enumerate(hots):
            if i < len(m.group(4)):
                r['v_in_hot_%d' % h] = int(m.group(4)[i])
        rows[(r['conv'], r['qi'])] = r

R = [rows[k] for k in sorted(rows, key=lambda x: (int(x[0]), x[1]))]
json.dump(R, open('hot2_all.json', 'w'))
cnt = collections.Counter(r['conv'] for r in R)
print('logs %s' % files)
print('recovered %d items, %d conversation clusters' % (len(R), len(cnt)))
print('per cluster: %s' % dict(sorted(cnt.items(), key=lambda x: int(x[0]))))
print('hot-window sizes: %s' % hots)
for h in hots:
    n = sum(r.get('v_in_hot_%d' % h, 0) for r in R)
    print('  hot=%-2d  V inside the hot window, must drop %d items (%.1f%%)' % (h, n, 100 * n / max(len(R), 1)))

z = [abs(r['c5_0_%sbhot' % t] - r['c5_0_%sharv' % t]) for r in R for t in ('', 'p_')
     if 'c5_0_%sbhot' % t in r]
print('self-check 1  bhot == harv @ hot=0: max|d| = %.8f over %d pairs  (must be 0)' % (max(z), len(z)))

# self-check 2: the three pre-existing arms must be unchanged vs the kv_e2hot run
OLD = re.compile(r'^\[e2hot\] (\S+) q(\d+) \| (.*?) \| vhot=')
OC = re.compile(r'k(\d+)h(\d+) iso=(\S+) self=(\S+) harv=(\S+) p_iso=(\S+) p_self=(\S+) p_harv=(\S+)')
prev = {}
for f in sorted(glob.glob('hot_*.log')):
    for line in open(f, errors='ignore'):
        m = OLD.match(line)
        if not m:
            continue
        d = {}
        for c in OC.findall(m.group(3)):
            for a, v in zip(('iso', 'self', 'harv', 'p_iso', 'p_self', 'p_harv'), c[2:]):
                d['c%s_%s_%s' % (c[0], c[1], a)] = float(v)
        prev[(m.group(1), int(m.group(2)))] = d
n2 = bad2 = shared = 0
for r in R:
    p = prev.get((r['conv'], r['qi']))
    if not p:
        continue
    shared += 1
    for f, v in p.items():
        if f in r:
            n2 += 1
            bad2 += (r[f] != v)
if shared:
    print('self-check 2  the three old arms vs the previous kv_e2hot run: %d numbers / %d common items, %d mismatches  (must be 0)'
          % (n2, shared, bad2))
else:
    print('self-check 2  skipped (no hot_*.log from the previous run in this directory)')
