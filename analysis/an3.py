import json, math, statistics as st
D=json.load(open('e2_k.json')); R=D['rows']
S={(str(i['conv']),i['qi']):i for i in json.load(open('e2_sets.json'))['items']}
for r in R: r['_s']=S[(str(r['conv']),r['qi'])]
def lo(r,k,a,tag=''): return r['k%d_%s%s_lo'%(k,tag,a)]
def pg(r,k,a,tag=''): return r['k%d_%s%s_p'%(k,tag,a)]
def dd(r,k,tag=''):   return lo(r,k,'harv',tag)-lo(r,k,'self',tag)
K=5
for r in R:
    s=r['_s']; sv=r['serve'][:K]; g=s.get('gold_idx') or []
    r['_span']=max(sv)-min(sv); r['_first']=min(sv); r['_last']=max(sv)
    r['_nturns']=s['n_turns']; r['_recency']=(s['n_turns']-max(sv))/s['n_turns']
    r['_depth']=min(sv)/s['n_turns']
    r['_dgold']=min((abs(a-b) for a in sv for b in g), default=-1)
    r['_d']=dd(r,K)

print('=== 1 · do wins/losses change the final answer? (forced choice, p_gold>0.5 = correct) ===')
for k in (1,2,3,5):
    hw=sum(1 for r in R if pg(r,k,'harv')>.5 and pg(r,k,'self')<=.5)
    sw=sum(1 for r in R if pg(r,k,'self')>.5 and pg(r,k,'harv')<=.5)
    same=len(R)-hw-sw
    ah=sum(1 for r in R if pg(r,k,'harv')>.5); asf=sum(1 for r in R if pg(r,k,'self')>.5)
    print('  k=%d : harv right/self wrong %3d  |  self right/harv wrong %3d  |  same verdict %3d   (overall accuracy harv %.3f  self %.3f)'%(k,hw,sw,same,ah/len(R),asf/len(R)))

print('\n=== 2 · by question type (cat): harv-self at k=5 ===')
cats={}
for r in R: cats.setdefault(r['cat'],[]).append(r['_d'])
NAME={1:'multi-hop',2:'temporal',3:'open/inference',4:'single-hop',5:'adversarial (no answer)'}
for c in sorted(cats):
    v=cats[c]; pos=sum(1 for x in v if x>0); neg=sum(1 for x in v if x<0)
    print('  cat%d %-10s n=%3d  mean %+7.4f  med %+6.3f  win %3d loss %3d'%(c,NAME.get(c,''),len(v),st.mean(v),st.median(v),pos,neg))

print('\n=== 3 · by conversation (cluster): harv-self at k=5 ===')
cv={}
for r in R: cv.setdefault(r['conv'],[]).append(r['_d'])
for c in sorted(cv,key=lambda x:int(x)):
    v=cv[c]; print('  conv %-3s n=%3d  mean %+7.4f  med %+6.3f  win %3d loss %3d'%(c,len(v),st.mean(v),st.median(v),sum(1 for x in v if x>0),sum(1 for x in v if x<0)))
gm=[st.mean(v) for v in cv.values()]
print('  --> cluster means %d positive / %d negative ; range [%+0.3f,%+0.3f]'%(sum(1 for x in gm if x>0),sum(1 for x in gm if x<0),min(gm),max(gm)))

def pear(x,y):
    mx=st.mean(x);my=st.mean(y)
    sx=math.sqrt(sum((a-mx)**2 for a in x));sy=math.sqrt(sum((b-my)**2 for b in y))
    return float('nan') if sx==0 or sy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)

print('\n=== 4 · profile of good vs bad cases (k=5) ===')
W=[r for r in R if r['_d']>1.0]; L=[r for r in R if r['_d']<-1.0]; T=[r for r in R if abs(r['_d'])<=1.0]
print('  harv clearly wins (Δ>+1)  n=%d   |   self clearly wins (Δ<-1)  n=%d   |   roughly tied  n=%d'%(len(W),len(L),len(T)))
FE=[('served-row span','_span'),('earliest served row','_first'),('latest served row','_last'),
    ('conversation turns','_nturns'),('distance from end (frac)','_recency'),('start depth (frac)','_depth'),
    ('nearest distance to gold','_dgold'),('usable V rows nV_used','nV_used')]
print('  %-24s %10s %10s %10s   corr(with Δ)'%('feature','harv wins','self wins','tied'))
for lab,f in FE:
    a=[r[f] for r in W]; b=[r[f] for r in L]; c=[r[f] for r in T]
    rr=pear([r[f] for r in R],[r['_d'] for r in R])
    print('  %-24s %10.1f %10.1f %10.1f   %+0.3f'%(lab,st.mean(a),st.mean(b),st.mean(c),rr))
for lab,f in (('no-context prior r_none_lo','r_none_lo'),):
    a=[r[f] for r in W]; b=[r[f] for r in L]; c=[r[f] for r in T]
    print('  %-24s %10.2f %10.2f %10.2f   %+0.3f'%(lab,st.mean(a),st.mean(b),st.mean(c),pear([r[f] for r in R],[r['_d'] for r in R])))
a=[lo(r,K,'iso') for r in W]; b=[lo(r,K,'iso') for r in L]; c=[lo(r,K,'iso') for r in T]
print('  %-24s %10.2f %10.2f %10.2f   %+0.3f'%('iso arm log-odds',st.mean(a),st.mean(b),st.mean(c),pear([lo(r,K,'iso') for r in R],[r['_d'] for r in R])))
