"""natcost.py -- read-out of the cost axis (kv_e2cost.py output). Latency microbenchmark, one A800."""
import json, statistics as st, collections
C = json.load(open('e2cost.json'))
rows = C['rows']; W = C['write']
print('reps=%d  k=%d  hots=%s  items=%d  convs=%s' % (C['reps'], C['k'], C['hots'], len(rows)//len(C['hots']), sorted({r['conv'] for r in rows})))
for w in W:
    print('WRITE conv %s: %d turns, %d tok, full prefill %.0f ms (%.1f ms/1k tok)' % (w['conv'], w['turns'], w['tokens'], w['t_write_ms'], 1000*w['t_write_ms']/w['tokens']))
ARMS = ('tx', 'iso', 'self', 'harv', 'bhot')
print('\n%-4s %-7s %-7s | %s' % ('hot', 'ctx_tok', 'q_tok', '  '.join('%-14s' % a for a in ARMS)))
print('%-4s %-7s %-7s | %s' % ('', '', '', '  '.join('%-14s' % 'build/TTFT ms' for a in ARMS)))
for h in C['hots']:
    u = [r for r in rows if r['hot'] == h]
    med = lambda f: st.median([r[f] for r in u])
    cells = []
    for a in ARMS:
        b = med('build_%s_ms' % a); t = med('ttft_%s_ms' % a)
        cells.append('%5s/%5.0f' % ('-' if a == 'tx' else '%.0f' % b, t))
    print('%-4d %-7.0f %-7.0f | %s' % (h, med('n_ctx_tok'), med('n_q_tok'), '  '.join('%-14s' % c for c in cells)))
print('\nTTFT deltas (median of per-item differences, ms):')
for h in C['hots']:
    u = [r for r in rows if r['hot'] == h]
    d = lambda a, b: st.median([r['ttft_%s_ms' % a] - r['ttft_%s_ms' % b] for r in u])
    print('  hot=%2d  self-bhot %+6.1f   self-harv %+6.1f   tx-bhot %+6.1f   iso-self %+6.1f   bhot-harv %+6.1f'
          % (h, d('self', 'bhot'), d('self', 'harv'), d('tx', 'bhot'), d('iso', 'self'), d('bhot', 'harv')))
print('\nper-token price of the re-encode (self build ms / 1k ctx tok):')
for h in C['hots']:
    u = [r for r in rows if r['hot'] == h]
    print('  hot=%2d  %.0f ms/1k' % (h, st.median([1000*r['build_self_ms']/r['n_ctx_tok'] for r in u])))
