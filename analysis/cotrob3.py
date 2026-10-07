import json, math, statistics as st, sys, collections
sys.path.insert(0,'.'); from wcrfast import W
R=json.load(open('cot_all.json'))
CONV=sorted({r['conv'] for r in R}); IDX={c:i for i,c in enumerate(CONV)}
acc=lambda r,pre,a: 1.0 if r[pre+a]>0 else 0.0
def wcr(rows,d):
    g=[IDX[r['conv']] for r in rows]; w=W(d,g,G=len(CONV)); return w.mean,w.p(),w.ci()
def con(rows,A,B,pre='c_'): return [acc(r,pre,A)-acc(r,pre,B) for r in rows]

print('n=%d  clusters=%d  per cluster=%s'%(len(R),len(CONV),dict(sorted(collections.Counter(r['conv'] for r in R).items()))))

print('\n### blind filter: drop by no-CoT saturation (a pre-treatment variable)')
sub=[r for r in R if all(abs(r['n_'+a])<15 for a in ('iso','self','harv'))]
print('  n=%d (%.0f%% kept), clusters %d'%(len(sub),100*len(sub)/len(R),len({r['conv'] for r in sub})))
for A,B in (('self','iso'),('harv','iso'),('harv','self'),('p_self','p_iso'),('p_harv','p_iso')):
    m,p,ci=wcr(sub,con(sub,A,B)); print('  %-14s %+7.4f  p=%.4f  [%+.3f,%+.3f]'%('%s-%s'%(A,B),m,p,ci[0],ci[1]))

print('\n### non-blind filter (for contrast: by the saturation of CoT itself)')
sub2=[r for r in R if all(abs(r['c_'+a])<15 for a in ('iso','self','harv'))]
for A,B in (('self','iso'),('harv','self')):
    m,p,ci=wcr(sub2,con(sub2,A,B)); print('  %-14s %+7.4f  p=%.4f  (n=%d)'%('%s-%s'%(A,B),m,p,len(sub2)))

print('\n### leave one cluster out (self-iso, CoT, full sample)')
res=[]
for c in CONV:
    rows=[r for r in R if r['conv']!=c]
    g=[IDX[r['conv']] for r in rows]; w=W(con(rows,'self','iso'),g,G=len(CONV)-1)
    res.append((c,w.mean,w.p(),len(rows)))
    print('  drop cluster %s (n=%d): %+0.4f  p=%.4f'%(c,len(rows),w.mean,w.p()))
print('  => point-estimate range [%+0.4f, %+0.4f]  max p=%.4f'%(min(x[1] for x in res),max(x[1] for x in res),max(x[2] for x in res)))

print('\n### per-cluster self-iso (CoT)')
sg=[]
for c in CONV:
    rows=[r for r in R if r['conv']==c]; d=con(rows,'self','iso'); m=st.fmean(d); sg.append(m)
    print('  cluster %s n=%2d: %+0.4f'%(c,len(rows),m))
print('  signs: %d positive / %d negative / %d zero'%(sum(1 for x in sg if x>1e-9),sum(1 for x in sg if x<-1e-9),sum(1 for x in sg if abs(x)<=1e-9)))

print('\n### three-cell split (CoT, self vs iso)')
cells=collections.Counter()
for r in R:
    i,s=acc(r,'c_','iso'),acc(r,'c_','self')
    if i==0 and s==1: cells['iso wrong->self right']+=1
    elif i==1 and s==0: cells['iso right->self wrong']+=1
print('  %s'%dict(cells))
