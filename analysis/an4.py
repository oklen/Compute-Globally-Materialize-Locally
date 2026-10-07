import json, math, statistics as st, itertools
D=json.load(open('e2_k.json')); R=D['rows']
def pgv(r,k,a,tag=''): return r['k%d_%s%s_p'%(k,tag,a)]
def lov(r,k,a,tag=''): return r['k%d_%s%s_lo'%(k,tag,a)]
CONV=sorted({r['conv'] for r in R},key=lambda x:int(x)); G=len(CONV)
IDX={c:i for i,c in enumerate(CONV)}
SIGNS=list(itertools.product([-1,1],repeat=G))
def wcr(d,g):
    """null-imposed wild cluster bootstrap-t, Rademacher, enumerate all 2^G"""
    n=len(d); m=st.mean(d)
    def tstat(v):
        mm=st.mean(v); byg={}
        for x,gg in zip(v,g): byg.setdefault(gg,[]).append(x-mm)
        s=sum((sum(z))**2 for z in byg.values())
        se=math.sqrt(s)/n
        return (mm/se if se>0 else 0.0), se
    t,se=tstat(d)
    cnt=0
    for w in SIGNS:
        v=[w[gg]*x for x,gg in zip(d,g)]   # H0: mu=0 -> restricted residual = d
        tt,_=tstat(v)
        if abs(tt)>=abs(t)-1e-12: cnt+=1
    return m, se, t, cnt/len(SIGNS)
g=[IDX[r['conv']] for r in R]

print('--- self-check: reproduce the published harv-iso @k=5 ---')
d=[lov(r,5,'harv')-lov(r,5,'iso') for r in R]
m,se,t,p=wcr(d,g); print('   mean %+0.4f  SE %0.4f  t %+0.2f  WCR p %0.4f   (the record says +1.0016 / p=.0020)'%(m,se,t,p))

print('\n=== right/wrong ruler (p_gold>.5 scores 1): harv - self ===')
print('  %-6s %-8s %8s %8s %9s %9s %8s %8s'%('k','serve set','harv acc','self acc','Δacc','WCR p','harv wins','self wins'))
for k in (1,2,3,5):
    for tag,nm in (('','real'),('p_','placebo')):
        ah=[1.0 if pgv(r,k,'harv',tag)>.5 else 0.0 for r in R]
        asf=[1.0 if pgv(r,k,'self',tag)>.5 else 0.0 for r in R]
        d=[a-b for a,b in zip(ah,asf)]
        m,se,t,p=wcr(d,g)
        hw=sum(1 for x in d if x>0); sw=sum(1 for x in d if x<0)
        print('  k=%-4d %-8s %8.4f %8.4f %+9.4f %9.4f %8d %8d'%(k,nm,st.mean(ah),st.mean(asf),m,p,hw,sw))

print('\n=== reference: self - iso and harv - iso on the same ruler ===')
print('  %-6s %-10s %9s %9s'%('k','contrast','Δacc','WCR p'))
for k in (1,2,3,5):
    for a,b in (('self','iso'),('harv','iso')):
        d=[(1.0 if pgv(r,k,a)>.5 else 0.0)-(1.0 if pgv(r,k,b)>.5 else 0.0) for r in R]
        m,se,t,p=wcr(d,g)
        print('  k=%-4d %-10s %+9.4f %9.4f'%(k,a+'-'+b,m,p))
