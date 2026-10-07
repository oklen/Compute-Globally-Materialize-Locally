"""densres5.py -- why HotpotQA `supporting_facts` do not predict leave-one-out evict on the stated layer.

Reads dens.json (kv_dens.py output, 247 trajectories) and stratifies the sup-vs-other evict comparison
by (1) the total tool_out contribution D = full - frame, (2) whether the sup sentence contains the gold
(answer sentence) or not (bridge sentence), (3) whether the gold appears in >= 2 sentences (duplicate
retrievals).  Prints the case items whose HotpotQA text / trajectory notes are quoted at the bottom
(taken once from the HotpotQA dev file and the kv_traj output traj.json).

    python3 kvmemory/e2k_analysis/densres5.py > kvmemory/e2k_analysis/densres5_out.txt
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'dens.json')))
rows = [r for r in d['rows'] if 'stated_sentence' in r]
print('rows with stated layer', len(rows), '| traj', d.get('traj'))


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if a > b else (0.5 if a == b else 0.0)
    return w / (len(pos) * len(neg))


def summ(units, label):
    sp = [u['evict'] for u in units if u['is_sup'] == 1]
    ns = [u['evict'] for u in units if u['is_sup'] == 0]
    print('  %-46s sup %+.2f (med %+.2f, n=%d) | other %+.2f (med %+.2f, n=%d) | AUC %.3f | sup>1nat %.0f%%  other>1nat %.0f%%' % (
        label, st.mean(sp), st.median(sp), len(sp), st.mean(ns), st.median(ns), len(ns), auc(sp, ns),
        100 * sum(x > 1 for x in sp) / len(sp), 100 * sum(x > 1 for x in ns) / len(ns)))


# same filter as densres.py: prior not saturated, |none| < 3
R = [r for r in rows if abs(r['stated_none']) < 3]
print('after |none|<3 filter', len(R))
for r in R:
    r['D'] = r['stated_full'] - r['stated_frame']          # total tool_out contribution
    r['n_gold'] = sum(u['has_gold'] for u in r['stated_sentence'])
allu = [u for r in R for u in r['stated_sentence']]
print('\n== overall (replicates densres.py) ==')
summ(allu, 'all items')
Ds = sorted(r['D'] for r in R)
print('\n== total tool_out contribution D = full - frame (stated layer) ==')
print('  median D %.2f | D<0.5: %d | D<2: %d | D>=2: %d | D>=5: %d  (of %d)' % (
    st.median(Ds), sum(x < .5 for x in Ds), sum(x < 2 for x in Ds), sum(x >= 2 for x in Ds), sum(x >= 5 for x in Ds), len(Ds)))
print('  notes state v (stated_stated=1): %d / %d ; note_states_gold=1: %d' % (
    sum(r['stated_stated'] for r in R), len(R), sum(r['note_states_gold'] for r in R)))
print('\n== stratified by D ==')
for lo, hi, lab in [(-99, 2, 'D<2 (docs do not matter)'), (2, 5, '2<=D<5'), (5, 999, 'D>=5 (docs matter)')]:
    us = [u for r in R if lo <= r['D'] < hi for u in r['stated_sentence']]
    if us:
        summ(us, lab + ' items=%d' % sum(lo <= r['D'] < hi for r in R))
print('\n== stratified by whether the notes/answer state v ==')
for flag in (0, 1):
    us = [u for r in R if r['stated_stated'] == flag for u in r['stated_sentence']]
    if us:
        summ(us, 'stated_stated=%d items=%d medD=%.2f' % (flag, sum(r['stated_stated'] == flag for r in R),
                                                         st.median([r['D'] for r in R if r['stated_stated'] == flag])))
print('\n== sup sentences split: contains gold (answer sentence) vs not (bridge/other sup) ==')
sg = [u['evict'] for u in allu if u['is_sup'] == 1 and u['has_gold'] == 1]
sb = [u['evict'] for u in allu if u['is_sup'] == 1 and u['has_gold'] == 0]
ng = [u['evict'] for u in allu if u['is_sup'] == 0 and u['has_gold'] == 1]
nn = [u['evict'] for u in allu if u['is_sup'] == 0 and u['has_gold'] == 0]
for lab, x in [('sup & has_gold', sg), ('sup & no gold (bridge)', sb), ('non-sup & has_gold (redundant copy)', ng), ('non-sup & no gold', nn)]:
    print('  %-38s mean %+.2f  med %+.2f  >1nat %.0f%%  n=%d' % (lab, st.mean(x), st.median(x), 100 * sum(v > 1 for v in x) / len(x), len(x)))
print('  AUC(has_gold as predictor) = %.3f ; AUC(is_sup | D>=5 items) = %.3f' % (
    auc([u['evict'] for u in allu if u['has_gold']], [u['evict'] for u in allu if not u['has_gold']]),
    auc([u['evict'] for r in R if r['D'] >= 5 for u in r['stated_sentence'] if u['is_sup'] == 1],
        [u['evict'] for r in R if r['D'] >= 5 for u in r['stated_sentence'] if u['is_sup'] == 0])))
print('\n== redundancy inside the docs: gold appears in k sentences ==')
for k, lab in [(1, 'n_gold=1'), (2, 'n_gold>=2')]:
    sel = [r for r in R if (r['n_gold'] == 1 if k == 1 else r['n_gold'] >= 2)]
    if not sel:
        continue
    ev = [u['evict'] for r in sel for u in r['stated_sentence'] if u['has_gold']]
    print('  %-10s items=%d  medD=%.2f  gold-sentence evict mean %+.2f med %+.2f' % (lab, len(sel), st.median([r['D'] for r in sel]), st.mean(ev), st.median(ev)))
print('  items with n_gold=0 (gold not literally in any tool_out sentence): %d' % sum(r['n_gold'] == 0 for r in R))


def show(r, tag):
    print('\n[%s] id=%s  D=%.2f  none=%+.2f  full=%+.2f  notes_state_v=%d  n_gold_sents=%d' % (tag, r['id'], r['D'], r['stated_none'], r['stated_full'], r['stated_stated'], r['n_gold']))
    print('   q: %s' % r['q1'][:140])
    print('   v: %s  (neg: %s)   sup=%s' % (r['v'], r['stated_neg'], r['sup']))
    us = sorted(r['stated_sentence'], key=lambda u: -u['evict'])
    for u in us[:4]:
        print('   evict %+6.2f  sup=%d gold=%d src=%d  %s #%s (%d tok)' % (u['evict'], u['is_sup'], u['has_gold'], u['is_src'], u['title'], u['sent_id'], u['n']))
    for u in [u for u in r['stated_sentence'] if u['is_sup'] == 1 and u not in us[:4]]:
        print('   evict %+6.2f  sup=%d gold=%d src=%d  %s #%s (%d tok)   <- other sup' % (u['evict'], u['is_sup'], u['has_gold'], u['is_src'], u['title'], u['sent_id'], u['n']))


R.sort(key=lambda r: -r['D'])
print('\n########## cases: docs matter most (largest D) ##########')
for r in R[:3]:
    show(r, 'D-top')
print('\n########## cases: docs do not matter (D<0.3, typical) ##########')
for r in [r for r in R if r['D'] < .3][:3]:
    show(r, 'D-zero')
print('\n########## cases: a NON-sup sentence with evict > 3 ##########')
cnt = 0
for r in R:
    if [u for u in r['stated_sentence'] if u['is_sup'] == 0 and u['evict'] > 3]:
        show(r, 'nonsup-heavy'); cnt += 1
    if cnt >= 3:
        break

CASE_TEXT = """
########## case text (from hotpot_dev_distractor.jsonl and the kv_traj output traj.json) ##########
===== 5a8be7735542997f31a41df2 | What conference does the team that drafted Derrick Hamilton in 2004 belong to?
answer: National Football Conference (NFC) West
  SUP [Derrick Hamilton #0] Derrick T. Hamilton (born November 30, 1981) is a former Canadian football wide receiver.
  SUP [Derrick Hamilton #1]  He was drafted by the San Francisco 49ers in the third round of the 2004 NFL Draft.
  SUP [San Francisco 49ers #1]  They compete in the National Football League (NFL) as a member of the league's National Football Conference (NFC) West division.
  plan: Think: I need to find out which team drafted Derrick Hamilton in 2004 and then determine the conference that team belongs to.
  tool_in: SEARCH: who drafted Derrick Hamilton in 2004
  tool_out titles: ['Derrick Hamilton', 'Derrick Hamilton (basketball)']
  scratchpad: NOTE: Derrick Hamilton was drafted by the San Francisco 49ers in the 2004 NFL Draft.
  tool_in: SEARCH: which conference does the San Francisco 49ers belong to
  tool_out titles: ['San Francisco 49ers', 'Derrick Hamilton']
  scratchpad: NOTE: The San Francisco 49ers belong to the National Football Conference (NFC).
  answer: ANSWER: National Football Conference (NFC)
  -> notes hold the bridge (49ers) and 'NFC' but not 'West': only the 49ers #1 sentence carries the readout (+16.75);
     bridge sentences +1.5/+0.75; the second retrieved copy of the Derrick Hamilton passage -1.0/-0.75.
===== 5a7502715542996c70cfae74 | Which Sudanese-born Australian professional basketball player for the Milwaukee Bucks exemplifies ...
answer: Thon Marial Maker
  SUP [NBA high school draftees #0] ... drafted to the NBA straight out of high school ...
  SUP [NBA high school draftees #3] ... or even hold themselves back a year in high school before declaring for the draft, such as the case with Thon Maker.
  SUP [Thon Maker #0] Thon Marial Maker (born 25 February 1997) is a Sudanese-born Australian professional basketball player for the Milwaukee Bucks ...
  tool_out titles: ['Thon Maker', 'Matthew Dellavedova']   then   ['Thon Maker', 'NBA high school draftees']
  scratchpad: NOTE: Thon Maker is a Sudanese-born Australian player for the Milwaukee Bucks ...   (full name never written)
  -> D=14.6 but the answer sentence is retrieved twice: leave-one-out gives +0.88 and 0.00 for the two copies.
===== 5a886b7555429938390d3f4d | When did the party inspired by Cacareco begin in Canada?
answer: 1960s
  SUP [Cacareco #4]  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  SUP [Rhinoceros Party of Canada (1963-93) #0] The Rhinoceros Party ... was a registered political party in Canada from the 1960s to the 1990s.
  tool_out titles: ['Cacareco', 'Revolutionary Communist Party of Canada']
  scratchpad: NOTE: The Rhinoceros Party of Canada was inspired by Cacareco's candidacy and was founded in 1999.
  answer: ANSWER: 1999   (wrong)
  -> the official sup passage 'Rhinoceros Party' was never retrieved; readout leans on Cacareco #0 (+3.5) and a distractor containing '1960s' (+1.75).
===== 5ae826675542997ec2727733 | The Heart Never Lies is an album by the author of a state ballad entitled what?
answer: The Land of Enchantment
  SUP [Michael Martin Murphey #3]  Murphey is also the author of New Mexico's state ballad, "The Land of Enchantment".
  scratchpad: NOTE: Shania Twain is not the author of the state ballad; the author is Michael Martin Murphey, and the state ballad is entitled "The Land of Enchantment".
  answer: ANSWER: The Land of Enchantment
  -> notes state the answer verbatim: D=0.25, every sentence |evict| <= 0.4, the gold-bearing sup sentence -0.38.
"""
print(CASE_TEXT)
