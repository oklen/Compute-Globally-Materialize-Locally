import json, math, statistics as st, itertools
D=json.load(open('e2_k.json')); R=D['rows']
def P(r,k,a,t=''): return r['k%d_%s%s_p'%(k,t,a)]
def L(r,k,a,t=''): return r['k%d_%s%s_lo'%(k,t,a)]
CONV=sorted({r['conv'] for r in R},key=lambda x:int(x)); G=len(CONV); IDX={c:i for i,c in enumerate(CONV)}
SIGNS=list(itertools.product([-1,1],repeat=G))
def wcr(d,g):
    n=len(d); m=st.mean(d)
    def T(v):
        mm=st.mean(v); byg={}
        for x,gg in zip(v,g): byg.setdefault(gg,[]).append(x-mm)
        s=sum((sum(z))**2 for z in byg.values()); se=math.sqrt(s)/n
        return (mm/se if se>0 else 0.0), se
    t,se=T(d); c=sum(1 for w in SIGNS if abs(T([w[gg]*x for x,gg in zip(d,g)])[0])>=abs(t)-1e-12)
    return m,se,c/len(SIGNS)
g=[IDX[r['conv']] for r in R]
acc=lambda r,k,a,t='': 1.0 if P(r,k,a,t)>.5 else 0.0

print('=== difference-in-differences: does the splice cost appear only on the rows the retriever actually selected? ===')
print('   DiD = (harv-self on real serve sets) - (harv-self on placebo serve sets), paired per item')
print('   %-6s %12s %10s %12s %10s'%('k','DiD acc','WCR p','DiD log-odds','WCR p'))
for k in (1,2,3,5):
    da=[(acc(r,k,'harv')-acc(r,k,'self'))-(acc(r,k,'harv','p_')-acc(r,k,'self','p_')) for r in R]
    dl=[(L(r,k,'harv')-L(r,k,'self'))-(L(r,k,'harv','p_')-L(r,k,'self','p_')) for r in R]
    ma,_,pa=wcr(da,g); ml,_,pl_=wcr(dl,g)
    print('   k=%-4d %+12.4f %10.4f %+12.4f %10.4f'%(k,ma,pa,ml,pl_))

print('\n=== k=2/3/5 averaged per item into one number (k=1 excluded: there self == iso) ===')
for nm,tag in (('real','') ,('placebo','p_')):
    da=[st.mean([acc(r,k,'harv',tag)-acc(r,k,'self',tag) for k in (2,3,5)]) for r in R]
    m,se,p=wcr(da,g); print('   %-6s harv-self acc %+0.4f  (SE %0.4f)  WCR p %0.4f'%(nm,m,se,p))
da=[st.mean([(acc(r,k,'harv')-acc(r,k,'self'))-(acc(r,k,'harv','p_')-acc(r,k,'self','p_')) for k in (2,3,5)]) for r in R]
m,se,p=wcr(da,g); print('   DiD                   %+0.4f  (SE %0.4f)  WCR p %0.4f'%(m,se,p))

print('\n=== where in the distribution does this cost fall? (k=5, real serve sets) ===')
d=[L(r,5,'harv')-L(r,5,'self') for r in R]; d.sort()
n=len(d)
for lab,sel in (('all',d),('trim 10% each end',d[int(.1*n):n-int(.1*n)]),('middle, |Δ|<2',[x for x in d if abs(x)<2])):
    v=sorted(sel); pos=sum(1 for x in v if x>0); neg=sum(1 for x in v if x<0)
    print('   %-16s n=%3d  mean %+0.4f  median %+0.4f  win %3d loss %3d'%(lab,len(v),st.mean(v),st.median(v),pos,neg))
print('   --> quantiles: ', ' '.join('%d%%:%+0.2f'%(q,d[int(q/100*n)]) for q in (5,25,50,75,95)))

print('\n=== robustness: drop items saturated in either arm (p<=1e-4 or >=0.9999) ===')
for k in (2,3,5):
    sub=[r for r in R if all(1e-4<P(r,k,a)<0.9999 for a in ('harv','self'))]
    gg=[IDX[r['conv']] for r in sub]
    da=[acc(r,k,'harv')-acc(r,k,'self') for r in sub]
    dl=[L(r,k,'harv')-L(r,k,'self') for r in sub]
    m,_,p=wcr(da,gg); m2,_,p2=wcr(dl,gg)
    print('   k=%d  n=%3d  Δacc %+0.4f (p %0.4f)   Δlog-odds %+0.4f (p %0.4f)'%(k,len(sub),m,p,m2,p2))
