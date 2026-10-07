import json, math, statistics as st, sys
sys.path.insert(0,'.'); from wcrfast import W
R=json.load(open('cot_all.json'))
CONV=sorted({r['conv'] for r in R}); IDX={c:i for i,c in enumerate(CONV)}
g=[IDX[r['conv']] for r in R]
acc=lambda r,pre,a: 1.0 if r[pre+a]>0 else 0.0
def wcr(d):
    w=W(d,g,G=len(CONV)); return w.mean,w.p(),w.ci()
print('n=%d, %d clusters. WCR enumerates 2^%d=%d sign vectors; smallest attainable p = %.4f'
      %(len(R),len(CONV),len(CONV),2**len(CONV),2/2**len(CONV)))

print('\n=== per-arm accuracy ===')
print('  %-8s %8s %8s %8s'%('arm','no CoT','CoT','Δ'))
for a in ('iso','self','harv','p_iso','p_self','p_harv'):
    n=st.fmean([acc(r,'n_',a) for r in R]); c=st.fmean([acc(r,'c_',a) for r in R])
    print('  %-8s %8.3f %8.3f %+8.3f'%(a,n,c,c-n))
print('  %-8s %8s %8.3f'%('none','—',st.fmean([acc(r,'c_','none') for r in R])))

print('\n=== three contrasts · accuracy ruler · WCR (cluster = conversation) ===')
print('  %-14s | %-30s | %-30s'%('contrast','no CoT','CoT'))
for A,B in (('self','iso'),('harv','iso'),('harv','self'),
            ('p_self','p_iso'),('p_harv','p_iso'),('p_harv','p_self')):
    dn=[acc(r,'n_',A)-acc(r,'n_',B) for r in R]
    dc=[acc(r,'c_',A)-acc(r,'c_',B) for r in R]
    mn,pn,cn=wcr(dn); mc,pc,cc_=wcr(dc)
    print('  %-14s | %+7.4f p=%.4f [%+.3f,%+.3f] | %+7.4f p=%.4f [%+.3f,%+.3f]'
          %('%s-%s'%(A,B),mn,pn,cn[0],cn[1],mc,pc,cc_[0],cc_[1]))

print('\n=== key pairing: does CoT change the contrast itself? (per item: CoT diff - no-CoT diff) ===')
for A,B in (('harv','iso'),('harv','self'),('self','iso')):
    d=[(acc(r,'c_',A)-acc(r,'c_',B))-(acc(r,'n_',A)-acc(r,'n_',B)) for r in R]
    m,p,ci=wcr(d)
    print('  %-12s Δ(CoT-noCoT) = %+0.4f  WCR p=%.4f  CI [%+0.4f,%+0.4f]'%('%s-%s'%(A,B),m,p,ci[0],ci[1]))

print('\n=== robustness: CoT contrasts after dropping saturated items (|lo|>=15 in any arm) ===')
sub=[r for r in R if all(abs(r['c_'+a])<15 for a in ('iso','self','harv'))]
gs=[IDX[r['conv']] for r in sub]
def wcr2(d):
    w=W(d,gs,G=len(CONV)); return w.mean,w.p()
print('  n=%d (%.0f%% kept), clusters %d'%(len(sub),100*len(sub)/len(R),len({r['conv'] for r in sub})))
for A,B in (('self','iso'),('harv','iso'),('harv','self'),('p_harv','p_iso')):
    m,p=wcr2([acc(r,'c_',A)-acc(r,'c_',B) for r in sub])
    print('  %-14s %+7.4f  p=%.4f'%('%s-%s'%(A,B),m,p))
