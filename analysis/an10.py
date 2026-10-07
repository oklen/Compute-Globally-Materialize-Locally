import json, math, statistics as st, itertools
D=json.load(open('e2_k.json')); R=D['rows']
def P(r,k,a,t=''): return r['k%d_%s%s_p'%(k,t,a)]
def L(r,k,a,t=''): return r['k%d_%s%s_lo'%(k,t,a)]
CONV=sorted({r['conv'] for r in R},key=lambda x:int(x)); IDX={c:i for i,c in enumerate(CONV)}
SIGNS=list(itertools.product([-1,1],repeat=len(CONV)))
def wcr(d,g):
    n=len(d); m=st.mean(d)
    def T(v):
        mm=st.mean(v); byg={}
        for x,gg in zip(v,g): byg.setdefault(gg,[]).append(x-mm)
        s=sum((sum(z))**2 for z in byg.values())
        return mm/(math.sqrt(s)/n) if s>0 else 0.0
    t=T(d); return m,sum(1 for w in SIGNS if abs(T([w[gg]*x for x,gg in zip(d,g)]))>=abs(t)-1e-12)/len(SIGNS)
g=[IDX[r['conv']] for r in R]
acc=lambda k,a,t='': sum(1 for r in R if P(r,k,a,t)>.5)/len(R)

print('=== three arms, two rulers (n=599, real serve sets) ===')
print('  %-4s | %25s | %25s'%('k','accuracy (forced choice)','mean log-odds (confidence)'))
print('  %-4s | %8s %8s %8s | %8s %8s %8s'%('','iso','self','harv','iso','self','harv'))
for k in (1,2,3,5):
    print('  k=%-2d | %8.4f %8.4f %8.4f | %+8.3f %+8.3f %+8.3f'
          %(k,acc(k,'iso'),acc(k,'self'),acc(k,'harv'),
            st.mean([L(r,k,'iso') for r in R]),st.mean([L(r,k,'self') for r in R]),st.mean([L(r,k,'harv') for r in R])))

print('\n=== three paired contrasts, two rulers (WCR p, cluster = conversation, all 1024 sign vectors) ===')
print('  %-12s | %-28s | %-28s'%('contrast','Δ accuracy','Δ log-odds'))
for a,b in (('self','iso'),('harv','iso'),('harv','self')):
    row=[]
    for k in (1,2,3,5):
        da=[(1. if P(r,k,a)>.5 else 0.)-(1. if P(r,k,b)>.5 else 0.) for r in R]
        dl=[L(r,k,a)-L(r,k,b) for r in R]
        ma,pa=wcr(da,g); ml,pl=wcr(dl,g); row.append((k,ma,pa,ml,pl))
    print('  %s - %s'%(a,b))
    for k,ma,pa,ml,pl in row:
        print('     k=%-2d      %+8.4f (p %.4f)          %+8.3f (p %.4f)'%(k,ma,pa,ml,pl))

print('\n=== k=2/3/5 averaged per item (one summary number) ===')
for a,b in (('self','iso'),('harv','iso'),('harv','self')):
    da=[st.mean([(1. if P(r,k,a)>.5 else 0.)-(1. if P(r,k,b)>.5 else 0.) for k in (2,3,5)]) for r in R]
    dl=[st.mean([L(r,k,a)-L(r,k,b) for k in (2,3,5)]) for r in R]
    ma,pa=wcr(da,g); ml,pl=wcr(dl,g)
    print('  %-11s  Δacc %+0.4f (p %.4f)   Δlog-odds %+0.3f (p %.4f)'%(a+'-'+b,ma,pa,ml,pl))

print('\n=== jitter vs signal across k ===')
print('  %-4s %16s %16s %10s'%('k','self-iso mean (signal)','harv-self sd (jitter)','SNR'))
for k in (1,2,3,5):
    sig=st.mean([L(r,k,'self')-L(r,k,'iso') for r in R]); noi=st.pstdev([L(r,k,'harv')-L(r,k,'self') for r in R])
    print('  k=%-2d %16.3f %16.3f %10.3f'%(k,sig,noi,sig/noi if noi else float('nan')))
