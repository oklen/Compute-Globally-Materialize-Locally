import json, statistics as st
D=json.load(open('e2_k.json')); R=D['rows']
S={(str(i['conv']),i['qi']):i for i in json.load(open('e2_sets.json'))['items']}
def P(r,k,a,t=''): return r['k%d_%s%s_p'%(k,t,a)]
def L(r,k,a,t=''): return r['k%d_%s%s_lo'%(k,t,a)]
K=5
for r in R:
    r['_d']=L(r,K,'harv')-L(r,K,'self'); r['_s']=S[(str(r['conv']),r['qi'])]
    r['_flip']= (1 if P(r,K,'harv')>.5 and P(r,K,'self')<=.5 else (-1 if P(r,K,'self')>.5 and P(r,K,'harv')<=.5 else 0))
def show(rs,title):
    print('\n'+'='*100); print(title); print('='*100)
    for r in rs:
        s=r['_s']
        print('  conv%s q%-3d cat%d  Δ=%+7.3f  flip=%s' % (r['conv'],r['qi'],r['cat'],r['_d'],
              {1:'harv right/self wrong',-1:'self right/harv wrong',0:'same verdict'}[r['_flip']]))
        print('    Q: %s'%s['question'][:96])
        print('    gold: %-38s  neg: %s'%(r['gold'][:38],r['neg'][:38]))
        print('    served rows %-28s conversation has %d turns, gold evidence at %s'%(str(r['serve'][:K]),s['n_turns'],s.get('gold_idx')))
        print('    p(gold):  none %.3f | iso %.3f | self %.3f | harv %.3f    log-odds: iso %+.2f self %+.2f harv %+.2f'
              %(r['r_none_p'],P(r,K,'iso'),P(r,K,'self'),P(r,K,'harv'),L(r,K,'iso'),L(r,K,'self'),L(r,K,'harv')))
srt=sorted(R,key=lambda r:-r['_d'])
show([r for r in srt if r['_flip']==1][:5],  'good cases · harv wins big and turns a wrong answer right (5 largest Δ)')
show([r for r in srt[::-1] if r['_flip']==-1][:5],'bad cases · self wins big and harv turns a right answer wrong (5 most negative Δ)')
show(srt[:3],'3 largest |Δ| (flipped or not) -- harv side')
show(srt[-3:],'3 largest |Δ| -- self side')

print('\n'+'='*100)
print('profile of flipped items')
print('='*100)
A=[r for r in R if r['_flip']==1]; B=[r for r in R if r['_flip']==-1]
print('  %-26s %14s %14s'%('','harv rescues (n=%d)'%len(A),'harv breaks (n=%d)'%len(B)))
for lab,f in (('item |Δ| log-odds',lambda r:abs(r['_d'])),
              ('iso arm p(gold)',lambda r:P(r,K,'iso')),
              ('no-context p(gold)',lambda r:r['r_none_p']),
              ('conversation turns',lambda r:r['_s']['n_turns']),
              ('served-row span',lambda r:max(r['serve'][:K])-min(r['serve'][:K])),
              ('usable V rows',lambda r:r['nV_used'])):
    print('  %-26s %14.3f %14.3f'%(lab,st.mean([f(r) for r in A]),st.mean([f(r) for r in B])))
print('  type mix  harv rescues:',{c:sum(1 for r in A if r['cat']==c) for c in sorted({r['cat'] for r in R})},
      ' harv breaks:',{c:sum(1 for r in B if r['cat']==c) for c in sorted({r['cat'] for r in R})})
