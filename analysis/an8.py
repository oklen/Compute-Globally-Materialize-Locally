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
def pear(x,y):
    mx=st.mean(x);my=st.mean(y); sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    return float('nan') if sx*sy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)

print('=== hypothesis: harv dilutes the effect of the served rows back toward the no-context prior ===')
print('   if so: (1) harv should sit closer to none than self does; (2) Δ should correlate strongly and positively with (none - self)\n')
print('   %-4s %14s %14s %13s %10s | %s'%('k','|harv-none|','|self-none|','diff (neg = harv closer)','WCR p','corr Δ vs (none-self)'))
for k in (1,2,3,5):
    hn=[abs(L(r,k,'harv')-r['r_none_lo']) for r in R]
    sn=[abs(L(r,k,'self')-r['r_none_lo']) for r in R]
    d=[a-b for a,b in zip(hn,sn)]; m,p=wcr(d,g)
    dd=[L(r,k,'harv')-L(r,k,'self') for r in R]
    ns=[r['r_none_lo']-L(r,k,'self') for r in R]
    print('   k=%-2d %14.3f %14.3f %13.3f %10.4f | %+0.3f'%(k,st.mean(hn),st.mean(sn),m,p,pear(dd,ns)))

print('\n=== shrinkage: least-squares regression of harv on (self, none) ===')
print('   harv ≈ a·self + b·none + c   -- b>0 and a<1 means shrinkage toward the prior')
for k in (2,3,5):
    X=[[L(r,k,'self'),r['r_none_lo'],1.0] for r in R]; y=[L(r,k,'harv') for r in R]
    n=3; A=[[sum(X[i][p]*X[i][q] for i in range(len(X))) for q in range(n)] for p in range(n)]
    b=[sum(X[i][p]*y[i] for i in range(len(X))) for p in range(n)]
    for i in range(n):
        pv=max(range(i,n),key=lambda r:abs(A[r][i])); A[i],A[pv]=A[pv],A[i]; b[i],b[pv]=b[pv],b[i]
        for r in range(n):
            if r!=i and A[i][i]!=0:
                f=A[r][i]/A[i][i]
                for c in range(n): A[r][c]-=f*A[i][c]
                b[r]-=f*b[i]
    sol=[b[i]/A[i][i] for i in range(n)]
    yh=[sum(X[i][p]*sol[p] for p in range(n)) for i in range(len(X))]
    ss=1-sum((y[i]-yh[i])**2 for i in range(len(y)))/sum((v-st.mean(y))**2 for v in y)
    print('   k=%d :  a(self) %+0.3f   b(none) %+0.3f   c %+0.3f   R² %0.3f'%(k,sol[0],sol[1],sol[2],ss))
    Xs=[[L(r,k,'self'),1.0] for r in R]
    d2=2; A2=[[sum(Xs[i][p]*Xs[i][q] for i in range(len(Xs))) for q in range(d2)] for p in range(d2)]
    b2=[sum(Xs[i][p]*y[i] for i in range(len(Xs))) for p in range(d2)]
    det=A2[0][0]*A2[1][1]-A2[0][1]*A2[1][0]
    a0=(b2[0]*A2[1][1]-A2[0][1]*b2[1])/det
    print('        (slope %+0.3f when regressing on self alone -- <1 means compressed overall)'%a0)

print('\n=== scale: is harv more spread out overall? ===')
for k in (1,2,3,5):
    print('   k=%-2d  sd(iso) %6.3f  sd(self) %6.3f  sd(harv) %6.3f   |  mean|lo| iso %5.2f self %5.2f harv %5.2f'
          %(k,st.pstdev([L(r,k,'iso') for r in R]),st.pstdev([L(r,k,'self') for r in R]),st.pstdev([L(r,k,'harv') for r in R]),
            st.mean([abs(L(r,k,'iso')) for r in R]),st.mean([abs(L(r,k,'self')) for r in R]),st.mean([abs(L(r,k,'harv')) for r in R])))
