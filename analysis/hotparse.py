"""hotparse.py -- rebuild the e2hot table from LOGS, not from the json.

Same reason as cotparse2: three worker reclaims during the CoT run destroyed the output json
every time and the logs were the only surviving copy.  kv_e2hot prints every number it measures
on one line per item, so the log IS the data.  Run this on a directory of `full.log` / `smoke.log`
files (`hot_*.log` after fetching) and it produces hot_all.json plus the instrument self-checks.
"""
import re, json, glob, sys, collections, statistics as st

ARMS = ("iso", "self", "harv", "p_iso", "p_self", "p_harv")
HEAD = re.compile(r'^\[e2hot\] (\S+) q(\d+) \| (.*?) \| vhot=(\d+) nu=(-?\d+) gold=')
CELL = re.compile(r'k(\d+)h(\d+) ' + ' '.join(r'%s=(\S+)' % a for a in ARMS))

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'hot_*.log'))
rows, seen = {}, set()
hots_seen = []
for f in files:
    for line in open(f, errors='ignore'):
        m = HEAD.match(line)
        if not m:
            continue
        conv, qi, body, vhot, nu = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
        cells = CELL.findall(body)
        if not cells:
            continue
        r = {'conv': conv, 'qi': qi, 'nunit': nu, 'vhot_bits': vhot}
        for c in cells:
            k, h = int(c[0]), int(c[1])
            for a, v in zip(ARMS, c[2:]):
                r['c%d_%d_%s' % (k, h, a)] = float(v)
        rows[(conv, qi)] = r          # later log wins; identical by construction anyway
        if not hots_seen:
            hots_seen = sorted({int(c[1]) for c in cells})
        # vhot bits are printed in sorted-HOTS order
        for i, h in enumerate(hots_seen):
            if i < len(vhot):
                r['v_in_hot_%d' % h] = int(vhot[i])

R = [rows[k] for k in sorted(rows, key=lambda x: (x[0], x[1]))]
json.dump(R, open('hot_all.json', 'w'))
cnt = collections.Counter(r['conv'] for r in R)
print('logs %s' % files)
print('recovered %d items, %d conversation clusters' % (len(R), len(cnt)))
print('per cluster: %s' % dict(sorted(cnt.items(), key=lambda x: int(x[0]))))
print('hot-window sizes: %s' % hots_seen)
for h in hots_seen:
    n = sum(r.get('v_in_hot_%d' % h, 0) for r in R)
    print('  hot=%-2d  V inside the hot window, must drop %d items (%.1f%%)' % (h, n, 100 * n / max(len(R), 1)))
# instrument self-check: k=1 hot=0 -> self and iso are the identical computation
z = [abs(r['c1_0_self'] - r['c1_0_iso']) for r in R if 'c1_0_self' in r]
if z:
    print('k=1 hot=0 null-by-construction: max|self-iso| = %.8f  (must be 0)' % max(z))
sat = 100 * st.fmean([abs(r['c5_0_' + a]) >= 15 for r in R for a in ('iso', 'self', 'harv')
                      if 'c5_0_' + a in r])
print('saturated(|lo|>=15, k5 hot0) %.1f%%' % sat)
