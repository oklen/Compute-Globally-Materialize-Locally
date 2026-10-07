import json, math, statistics as st, sys
sys.path.insert(0,'.'); from wcrfast import W
O=json.load(open('div3_join2.json'))
CONV=sorted({o['conv'] for o in O}); IDX={c:i for i,c in enumerate(CONV)}
for o in O: o['hi']=1 if o['ov']>=2 else 0
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
Xs=[[math.log(o['ntok']) for o in X],[math.log(o['dist']) for o in X],[1.0]*len(X)]
def resid(Y):
    co=ols(Y,Xs); return [Y[i]-sum(Xs[p][i]*co[p] for p in range(3)) for i in range(len(X))]
rL=resid([math.log(o['n_last']) for o in X])
rC=resid([math.log(o['n_rest']/max(o['ntok']-1,1)) for o in X])
# per-turn delimiter-minus-content difference: paired within a turn, so every turn-level common factor cancels
dLC=[a-b for a,b in zip(rL,rC)]
def clusters(vals):
    per={}
    for v,o in zip(vals,X): per.setdefault(o['conv'],[[],[]])[0 if o['hi'] else 1].append(v)
    return [(c,st.fmean(u[0])-st.fmean(u[1]),len(u[0]),len(u[1]))
            for c,u in per.items() if len(u[0])>=10 and len(u[1])>=10]
def sign_p(pos,n):
    from math import comb
    k=min(pos,n-pos); return min(1.0, 2*sum(comb(n,i) for i in range(0,k+1))/2**n)
def report(nm,vals):
    cl=clusters(vals)
    d=[x for _,x,_,_ in cl]; g=[IDX[c] for c,_,_,_ in cl]
    w=W(d,g,G=len(CONV)); ci=w.ci()
    pos=sum(1 for x in d if x>0)
    print('  %-26s clusters %2d | mean %+0.4f  WCR p %.4f  CI [%+0.4f,%+0.4f] | median %+0.4f  trimmed20%% %+0.4f | pos %d/%d sign test p %.4f'
          %(nm,len(d),w.mean,w.p(),ci[0],ci[1],st.median(d),
            st.fmean(sorted(d)[max(1,len(d)//5):len(d)-max(1,len(d)//5)]),pos,len(d),sign_p(pos,len(d))))
    return cl
print('=== cluster-level paired difference after residualization (>=2 shared words minus <2), three statistics side by side ===')
report('delimiter',rL); report('content token',rC)
cl=report('delimiter - content (paired)',dLC)
print('\n=== per-cluster detail: delimiter - content ===')
for c,x,nh,nl in sorted(cl,key=lambda z:-z[1]):
    print('    %-28s %+0.4f   (hi n=%4d, lo n=%5d)'%(c,x,nh,nl))
print('\n=== leave one cluster out: mean of delimiter-content after dropping each cluster ===')
d=[x for _,x,_,_ in cl]; g=[IDX[c] for c,_,_,_ in cl]
lo_,hi_=None,None
for i in range(len(d)):
    dd=d[:i]+d[i+1:]; gg=g[:i]+g[i+1:]
    m=W(dd,gg,G=len(CONV)).mean
    lo_=m if lo_ is None else min(lo_,m); hi_=m if hi_ is None else max(hi_,m)
print('    range [%+0.4f, %+0.4f]  (full sample %+0.4f)'%(lo_,hi_,st.fmean(d)))
