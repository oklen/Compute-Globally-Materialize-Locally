import json, math, statistics as st, sys
sys.path.insert(0,'.'); from wcrfast import W
O=json.load(open('div3_join2.json'))
CONV=sorted({o['conv'] for o in O}); IDX={c:i for i,c in enumerate(CONV)}
for o in O:
    o['ratio']=o['rel_last']/o['rel_rest'] if o['rel_rest']>0 else None
    o['npt_rest']=o['n_rest']/max(o['ntok']-1,1)
    o['hi']=1 if o['ov']>=2 else 0
def wcr(vals,convs):
    g=[IDX[c] for c in convs]; w=W(vals,g,G=len(CONV)); return w.mean,w.p(),w.ci()

print('sample: %d turns / %d conv / %d questions'%(len(O),len(CONV),len({(o['conv'],o['q']) for o in O})))
print('=== E · hi vs lo within length strata (tertiles) ===')
q=sorted(o['ntok'] for o in O); t1,t2=q[len(q)//3],q[2*len(q)//3]
def strat(f,nm):
    S=[o for o in O if f(o)]
    hi=[o for o in S if o['hi']]; lo=[o for o in S if not o['hi']]
    if len(hi)<15: print('  %-14s n(hi)=%d too few'%(nm,len(hi))); return
    row=[]
    for k in ('n_last','npt_rest','rel_last','rel_rest','ratio'):
        a=[o[k] for o in hi if o[k] is not None]; b=[o[k] for o in lo if o[k] is not None]
        row.append(st.fmean(a)/st.fmean(b) if b and st.fmean(b) else float('nan'))
    print('  %-14s n(hi)=%4d n(lo)=%5d | delimiter abs %.3f  content abs/tok %.3f | rel_last %.3f rel_rest %.3f | ratio %.3f'
          %(nm,len(hi),len(lo),row[0],row[1],row[2],row[3],row[4]))
strat(lambda o:o['ntok']<=t1,'short <=%d'%t1); strat(lambda o:t1<o['ntok']<=t2,'mid'); strat(lambda o:o['ntok']>t2,'long >%d'%t2)

print('\n=== F · residualized: regress log(delimiter abs drift) on log(length)+log(distance), then look at overlap ===')
X=[o for o in O if o['n_last']>0 and o['ntok']>1 and o['dist']>0]
import itertools
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
Y=[math.log(o['n_last']) for o in X]
Xs=[[math.log(o['ntok']) for o in X],[math.log(o['dist']) for o in X],[1.0]*len(X)]
co=ols(Y,Xs)
res=[Y[i]-sum(Xs[p][i]*co[p] for p in range(3)) for i in range(len(X))]
print('  coefficients: log(length) %+0.4f  log(distance) %+0.4f  const %+0.4f'%tuple(co))
hi=[r for r,o in zip(res,X) if o['hi']]; lo=[r for r,o in zip(res,X) if not o['hi']]
print('  residual mean  >=2 shared words %+0.4f (n=%d)   <2 %+0.4f (n=%d)   diff %+0.4f  (log scale, exp=%.3f)'
      %(st.fmean(hi),len(hi),st.fmean(lo),len(lo),st.fmean(hi)-st.fmean(lo),math.exp(st.fmean(hi)-st.fmean(lo))))
per={}
for r,o in zip(res,X): per.setdefault(o['conv'],[[],[]])[0 if o['hi'] else 1].append(r)
pairs=[(c,st.fmean(v[0])-st.fmean(v[1])) for c,v in per.items() if len(v[0])>=5 and len(v[1])>=5]
m,p,ci=wcr([x for _,x in pairs],[c for c,_ in pairs])
print('  cluster-level paired diff %+0.4f  WCR p %.4f  CI [%+0.4f,%+0.4f]  %d clusters, pos/neg %d/%d'
      %(m,p,ci[0],ci[1],len(pairs),sum(1 for _,x in pairs if x>0),sum(1 for _,x in pairs if x<0)))

print('\n=== G · the same residualization on content tokens (should be flat, else it is not delimiter-specific) ===')
Y2=[math.log(o['n_rest']/max(o['ntok']-1,1)) for o in X]
co2=ols(Y2,Xs); res2=[Y2[i]-sum(Xs[p][i]*co2[p] for p in range(3)) for i in range(len(X))]
h2=[r for r,o in zip(res2,X) if o['hi']]; l2=[r for r,o in zip(res2,X) if not o['hi']]
per2={}
for r,o in zip(res2,X): per2.setdefault(o['conv'],[[],[]])[0 if o['hi'] else 1].append(r)
pr2=[(c,st.fmean(v[0])-st.fmean(v[1])) for c,v in per2.items() if len(v[0])>=5 and len(v[1])>=5]
m2,p2,ci2=wcr([x for _,x in pr2],[c for c,_ in pr2])
print('  residual diff %+0.4f (exp %.3f)   cluster-level %+0.4f  WCR p %.4f  CI [%+0.4f,%+0.4f]  pos/neg %d/%d'
      %(st.fmean(h2)-st.fmean(l2),math.exp(st.fmean(h2)-st.fmean(l2)),m2,p2,ci2[0],ci2[1],
        sum(1 for _,x in pr2 if x>0),sum(1 for _,x in pr2 if x<0)))

print('\n=== H · per-cluster detail (delimiter residual diff) ===')
for c,x in sorted(pairs): print('    %-4s %+0.4f'%(c,x))
print('\n  sample: %d turns / %d conversations / %d questions; hi group %d'
      %(len(O),len(CONV),len({(o['conv'],o['q']) for o in O}),sum(o['hi'] for o in O)))
