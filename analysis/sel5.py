import json, math, statistics as st, sys
sys.path.insert(0,'.'); from wcrfast import W
O=json.load(open('div3_join2.json'))
CONV=sorted({o['conv'] for o in O}); IDX={c:i for i,c in enumerate(CONV)}
X=[o for o in O if o['n_last']>0 and o['n_rest']>0 and o['ntok']>1 and o['dist']>0]
def ols(Y,Xs):
    n=len(Xs[0]); k=len(Xs)
    A=[[sum(Xs[p][i]*Xs[q][i] for i in range(n)) for q in range(k)] for p in range(k)]
    b=[sum(Xs[p][i]*Y[i] for i in range(n)) for p in range(k)]
    for i in range(k):
        pv=max(range(i,k),key=lambda r:abs(A[r][i])); A[i],A[pv]=A[pv],A[i]; b[i],b[pv]=b[pv],b[i]
        for r in range(k):
            if r!=i and A[i][i]!=0:
                f=A[r][i]/A[i][i]
                for c in range(k): A[r][c]-=f*A[i][c]
                b[r]-=f*b[i]
    return [b[i]/A[i][i] for i in range(k)]
def mkres(withctrl=True):
    Xs=([[math.log(o['ntok']) for o in X],[math.log(o['dist']) for o in X],[1.0]*len(X)]
        if withctrl else [[1.0]*len(X)])
    def r(Y):
        co=ols(Y,Xs); return [Y[i]-sum(Xs[p][i]*co[p] for p in range(len(Xs))) for i in range(len(X))]
    return r
def test(vals,thr,minn=10):
    per={}
    for v,o in zip(vals,X): per.setdefault(o['conv'],[[],[]])[0 if o['ov']>=thr else 1].append(v)
    cl=[(c,st.fmean(u[0])-st.fmean(u[1])) for c,u in per.items() if len(u[0])>=minn and len(u[1])>=minn]
    if len(cl)<5: return None
    d=[x for _,x in cl]; g=[IDX[c] for c,_ in cl]
    w=W(d,g,G=len(CONV)); ci=w.ci()
    return len(d),w.mean,w.p(),ci[0],ci[1],sum(1 for x in d if x>0),st.median(d)

print('=== threshold robustness: >=T shared content words vs <T, paired quantity = delimiter residual - content residual ===')
print('  %-6s %5s %10s %8s %22s %8s %9s'%('thresh','G','mean','WCR p','95% CI','pos G','median'))
r=mkres(True)
rL=r([math.log(o['n_last']) for o in X]); rC=r([math.log(o['n_rest']/max(o['ntok']-1,1)) for o in X])
d=[a-b for a,b in zip(rL,rC)]
for thr in (1,2,3,4):
    t=test(d,thr)
    if not t: print('  >=%-4d (too few clusters)'%thr); continue
    print('  >=%-4d %5d %+10.4f %8.4f  [%+8.4f,%+8.4f] %6d/%-2d %+9.4f'%(thr,t[0],t[1],t[2],t[3],t[4],t[5],t[0],t[6]))

print('\n=== no residualization (raw log quantities)===')
r0=mkres(False)
rL0=r0([math.log(o['n_last']) for o in X]); rC0=r0([math.log(o['n_rest']/max(o['ntok']-1,1)) for o in X])
d0=[a-b for a,b in zip(rL0,rC0)]
for thr in (2,):
    t=test(d0,thr); print('  >=%-4d %5d %+10.4f %8.4f  [%+8.4f,%+8.4f] %6d/%-2d %+9.4f'%(thr,t[0],t[1],t[2],t[3],t[4],t[5],t[0],t[6]))

print('\n=== relative quantities rel_last / rel_rest instead (different ruler, same question)===')
Y1=[math.log(o['rel_last']) for o in X if o['rel_last']>0 and o['rel_rest']>0]
X2=[o for o in X if o['rel_last']>0 and o['rel_rest']>0]
Xs2=[[math.log(o['ntok']) for o in X2],[math.log(o['dist']) for o in X2],[1.0]*len(X2)]
def r2(Y):
    co=ols(Y,Xs2); return [Y[i]-sum(Xs2[p][i]*co[p] for p in range(3)) for i in range(len(X2))]
a=r2([math.log(o['rel_last']) for o in X2]); b=r2([math.log(o['rel_rest']) for o in X2])
dd=[u-v for u,v in zip(a,b)]
per={}
for v,o in zip(dd,X2): per.setdefault(o['conv'],[[],[]])[0 if o['ov']>=2 else 1].append(v)
cl=[(c,st.fmean(u[0])-st.fmean(u[1])) for c,u in per.items() if len(u[0])>=10 and len(u[1])>=10]
dq=[x for _,x in cl]; gq=[IDX[c] for c,_ in cl]; w=W(dq,gq,G=len(CONV)); ci=w.ci()
print('  >=2    %5d %+10.4f %8.4f  [%+8.4f,%+8.4f] %6d/%-2d %+9.4f'
      %(len(dq),w.mean,w.p(),ci[0],ci[1],sum(1 for x in dq if x>0),len(dq),st.median(dq)))

print('\n=== placebo: compute overlap with the words of another question (same turns, same drifts)===')
import collections
byconv=collections.defaultdict(list)
for o in X: byconv[o['conv']].append(o)
qs=collections.defaultdict(set)
for o in X: qs[o['conv']].add(o['q'])
STOP=set()
# each turn uses the overlap with *another question* in the same conversation: rotate ov by q within each conv
qmap={}
for c,S in qs.items():
    L=sorted(S); qmap.update({(c,q):L[(i+1)%len(L)] for i,q in enumerate(L)})
ovmap={(o['conv'],o['q'],o['j']):o['ov'] for o in X}
pl=[]
for v,o in zip(d,X):
    alt=qmap.get((o['conv'],o['q']))
    k=(o['conv'],alt,o['j'])
    pl.append((v,ovmap.get(k)))
per={}
for v,ov in pl:
    if ov is None: continue
per={}
for (v,ov),o in zip(pl,X):
    if ov is None: continue
    per.setdefault(o['conv'],[[],[]])[0 if ov>=2 else 1].append(v)
cl=[(c,st.fmean(u[0])-st.fmean(u[1])) for c,u in per.items() if len(u[0])>=10 and len(u[1])>=10]
if len(cl)>=5:
    dq=[x for _,x in cl]; gq=[IDX[c] for c,_ in cl]; w=W(dq,gq,G=len(CONV)); ci=w.ci()
    print('  placebo %5d %+10.4f %8.4f  [%+8.4f,%+8.4f] %6d/%-2d %+9.4f'
          %(len(dq),w.mean,w.p(),ci[0],ci[1],sum(1 for x in dq if x>0),len(dq),st.median(dq)))
else: print('  placebo: too few usable clusters (%d)'%len(cl))
