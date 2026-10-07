import json, math, statistics as st
D=json.load(open('e2_k.json'))
R=D['rows']; KS=D['ks']
print('n =',len(R),' arm =',D['arm'])
def col(k,a,tag=''):  return [r['k%d_%s%s_lo'%(k,tag,a)] for r in R]
def d(k,a,b,tag=''):  return [x-y for x,y in zip(col(k,a,tag),col(k,b,tag))]

def desc(v,lab):
    v=sorted(v); n=len(v)
    pos=sum(1 for x in v if x>1e-9); neg=sum(1 for x in v if x<-1e-9); zer=n-pos-neg
    tr=v[int(.05*n):n-int(.05*n)]
    print('%-16s mean %+7.4f  med %+7.4f  trim5%% %+7.4f  sd %6.3f | win %3d  lose %3d  tie %3d | p5 %+7.3f p95 %+7.3f | max|d| %6.2f'
          %(lab, st.mean(v), st.median(v), st.mean(tr), st.pstdev(v), pos, neg, zer, v[int(.05*n)], v[int(.95*n)-1], max(abs(x) for x in v)))

print('\n=== harv - self (real serve sets) ===')
for k in KS: desc(d(k,'harv','self'),'k=%d'%k)
print('\n=== harv - self (placebo serve sets, control) ===')
for k in KS: desc(d(k,'harv','self','p_'),'k=%d placebo'%k)
print('\n=== reference: self - iso (the "encoded together" channel) ===')
for k in KS: desc(d(k,'self','iso'),'k=%d'%k)
print('\n=== reference: harv - iso (PRIMARY) ===')
for k in KS: desc(d(k,'harv','iso'),'k=%d'%k)
