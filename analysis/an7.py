import json, math, statistics as st, itertools
D=json.load(open('e2_k.json')); R=D['rows']
S={(str(i['conv']),i['qi']):i for i in json.load(open('e2_sets.json'))['items']}
def P(r,k,a,t=''): return r['k%d_%s%s_p'%(k,t,a)]
def L(r,k,a,t=''): return r['k%d_%s%s_lo'%(k,t,a)]
CONV=sorted({r['conv'] for r in R},key=lambda x:int(x)); G=len(CONV); IDX={c:i for i,c in enumerate(CONV)}
SIGNS=list(itertools.product([-1,1],repeat=G))
def wcr(d,g):
    n=len(d); m=st.mean(d)
    def T(v):
        mm=st.mean(v); byg={}
        for x,gg in zip(v,g): byg.setdefault(gg,[]).append(x-mm)
        return ((mm/(math.sqrt(sum((sum(z))**2 for z in byg.values()))/n)) if sum((sum(z))**2 for z in byg.values())>0 else 0.0)
    t=T(d); c=sum(1 for w in SIGNS if abs(T([w[gg]*x for x,gg in zip(d,g)]))>=abs(t)-1e-12)
    return m,c/len(SIGNS)
K=5
def feat(serve,gold):
    sv=serve[:K]
    dn=[j-min(gold) for j in sv if any(j>gp for gp in gold)]
    downs=[min(j-gp for gp in gold if j>gp) for j in sv if any(j>gp for gp in gold)]
    return {'n_down':len(downs),'gap':(min(downs) if downs else 10**6),'any':1 if downs else 0}
ok=0
for r in R:
    s=S[(str(r['conv']),r['qi'])]; gi=s.get('gold_idx') or []
    r['_gi']=gi; r['_ok']=bool(gi); ok+=bool(gi)
    if gi:
        r['_f']=feat(r['serve'],gi); r['_fp']=feat(r['placebo'],gi)
    r['_d']=L(r,K,'harv')-L(r,K,'self')
    r['_a']=(1.0 if P(r,K,'harv')>.5 else 0.0)-(1.0 if P(r,K,'self')>.5 else 0.0)
    r['_dp']=L(r,K,'harv','p_')-L(r,K,'self','p_')
    r['_ap']=(1.0 if P(r,K,'harv','p_')>.5 else 0.0)-(1.0 if P(r,K,'self','p_')>.5 else 0.0)
Q=[r for r in R if r['_ok']]
print('items with a labelled gold-evidence position: %d / %d'%(len(Q),len(R)))

print('\n=== do the served rows include rows downstream of the gold evidence? (downstream = could see the gold evidence when encoded) ===')
for lab,key,ch in (('real serve sets','_f','_d'),('placebo serve sets','_fp','_dp')):
    print('  --- %s ---'%lab)
    for nm,sel in (('has downstream rows',lambda r:r[key]['any']==1),('all upstream',lambda r:r[key]['any']==0)):
        sub=[r for r in Q if sel(r)]
        if not sub: continue
        g=[IDX[r['conv']] for r in sub]
        m,p=wcr([r[ch] for r in sub],g)
        acc=('_a' if ch=='_d' else '_ap')
        ma,pa=wcr([r[acc] for r in sub],g)
        print('    %-8s n=%3d  Δlog-odds %+7.4f (p %.4f)   Δacc %+7.4f (p %.4f)'%(nm,len(sub),m,p,ma,pa))

print('\n=== bucketed by nearest downstream distance (real serve sets) ===')
B=[(1,3,'adjacent 1-3 turns'),(4,10,'4-10 turns'),(11,50,'11-50 turns'),(51,200,'51-200 turns'),(201,10**5,'>200 turns'),(10**6,10**7,'no downstream row')]
print('    %-14s %5s %12s %9s %12s %9s'%('downstream dist','n','Δlog-odds','WCR p','Δacc','WCR p'))
for a,b,nm in B:
    sub=[r for r in Q if a<=r['_f']['gap']<=b]
    if len(sub)<8: 
        print('    %-14s %5d   (n too small, skipped)'%(nm,len(sub))); continue
    g=[IDX[r['conv']] for r in sub]
    m,p=wcr([r['_d'] for r in sub],g); ma,pa=wcr([r['_a'] for r in sub],g)
    print('    %-14s %5d %+12.4f %9.4f %+12.4f %9.4f'%(nm,len(sub),m,p,ma,pa))

print('\n=== correlation: downstream row count / nearest downstream distance  vs  harv-self ===')
def pear(x,y):
    mx=st.mean(x);my=st.mean(y)
    sx=math.sqrt(sum((a-mx)**2 for a in x));sy=math.sqrt(sum((b-my)**2 for b in y))
    return float('nan') if sx==0 or sy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)
fin=[r for r in Q if r['_f']['gap']<10**5]
print('    downstream rows vs Δ      : %+0.3f  (n=%d)'%(pear([r['_f']['n_down'] for r in Q],[r['_d'] for r in Q]),len(Q)))
print('    log(nearest downstream dist) vs Δ: %+0.3f  (n=%d)'%(pear([math.log(max(r['_f']['gap'],1)) for r in fin],[r['_d'] for r in fin]),len(fin)))
