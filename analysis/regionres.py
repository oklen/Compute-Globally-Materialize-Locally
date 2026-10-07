"""regionres.py -- read-out of the seven-region carry experiment (kv_region.py).

Q1  per region r:  full - drop_r   (cost of evicting r)   full - tx_r  (KV-vs-text for r)
Q2  source tool_out counterfactual, keep-sets: allbutsrc / own (system+user+scratchpad+answer) /
    plan (system+user+plan) / srconly (system+user+s') ; d = lo_orig - lo_cf = what the keep-set
    carries of the source value.  placebo = swap a non-answer year in the source.
The contract's prediction: tool_out is load-bearing (dropping it
hurts, its KV beats text); scratchpad/answer carry what they STATE; plan/tool_in/system carry ~0.
For Q2 UNSTATED-in-downstream: own/plan keep-sets should carry ~0 (broad form) unless a downstream
region re-stated the value.
"""
import json, glob, sys, statistics as st, collections, math
sys.path.insert(0, '.')

J = None
for cand in ('region.json',):
    if glob.glob(cand):
        J = json.load(open(glob.glob(cand)[0])); break
if J is None:
    print('no region.json'); sys.exit(1)
rows = J['rows']
print('n=%d  skip=%s' % (len(rows), J.get('skip')))
REGIONS = ("system", "user", "plan", "tool_in", "tool_out", "scratchpad", "answer")

# token share
share = collections.Counter()
for r in rows:
    for k, v in (r.get('share') or {}).items():
        share[k] += v
tot = sum(share.values()) or 1
print('region token share: %s' % {k: round(share[k] / tot, 3) for k in REGIONS})
print('note states gold: %.2f   answer accuracy: %.2f'
      % (st.fmean(r.get('note_states_gold', 0) for r in rows), st.fmean(r.get('answer_correct', 0) for r in rows)))

def mean(key, sel=lambda r: True):
    v = [r[key] for r in rows if key in r and r[key] is not None and sel(r)]
    return (st.fmean(v), len(v)) if v else (float('nan'), 0)

for kind in ("stated", "unstated"):
    if not any(kind + "_full" in r for r in rows):
        continue
    use = [r for r in rows if kind + "_full" in r]
    fullm, _ = mean(kind + "_full", lambda r: kind + "_full" in r)
    print('\n' + '=' * 74 + '\n%s  n=%d  full level %+.2f  none %+.2f' % (kind, len(use), fullm, mean(kind + "_none")[0]))
    print('  %-11s %8s %8s %8s' % ('region', 'evict', 'KV-txt', 'n'))
    for rn in REGIONS:
        dk = kind + "_drop_" + rn; tk = kind + "_tx_" + rn
        if not any(dk in r for r in use):
            continue
        evict = st.fmean([r[kind + "_full"] - r[dk] for r in use if dk in r])   # full - drop = cost of evicting
        kvtx = st.fmean([r[tk] - r[dk] for r in use if tk in r and dk in r]) if any(tk in r for r in use) else float('nan')
        # KV-vs-text on the SAME served set: (drop others, keep r as KV) vs (keep r as text) = drop_r has r as KV? 
        # our arms: drop_r removes r entirely; tx_r re-encodes r alone. So KV-vs-text for r = (full - tx_r) - (full - drop_r)?
        # cleaner: full keeps r as in-situ KV; tx_r keeps r as text. KV-vs-text(r) = full - tx_r.
        kvtx = st.fmean([r[kind + "_full"] - r[tk] for r in use if tk in r])
        n = sum(1 for r in use if dk in r)
        print('  %-11s %+8.2f %+8.2f %8d' % (rn, evict, kvtx, n))
    print('  reading: evict = readout drop when the region is removed (larger = more load-bearing); KV-txt = full - text re-encoding of the region (>0 = KV is more faithful than text)')

# Q2 counterfactual keep-sets (source tool_out value swapped)
print('\n' + '=' * 74 + '\nQ2 swap the value in the source region (tool_out) -> how much source value each retained set carries')
for kn in ("allbutsrc", "own", "plan", "srconly", "placebo"):
    key = "q2_%s_d" % kn
    m, n = mean(key)
    if n:
        # paired significance vs 0
        d = [r[key] for r in rows if key in r and r[key] is not None]
        sd = st.pstdev(d) or 1e-9; p = math.erfc(abs((m / (sd / math.sqrt(len(d))))) / math.sqrt(2)) if len(d) > 1 else float('nan')
        print('  %-10s %+7.3f  p=%.4f  n=%d%s' % (kn, m, p, n, '  <<<' if p < .05 else ''))
print('  own=system+user+scratchpad+answer (the agent output); plan=system+user+plan; srconly=only the swapped source + frame (positive control, should be large); placebo=swap a non-answer year (should be ~0)')
# split by whether a downstream region re-stated the value (note_states_gold)
st_rows = [r for r in rows if r.get('note_states_gold') and 'q2_own_d' in r]
un_rows = [r for r in rows if not r.get('note_states_gold') and 'q2_own_d' in r]
for nm, rs in (('notes/answer state gold', st_rows), ('notes/answer do not state gold', un_rows)):
    if len(rs) >= 5:
        d = [r['q2_own_d'] for r in rs]; m = st.fmean(d)
        print('  own by statement: %-22s %+6.2f  n=%d' % (nm, m, len(rs)))
