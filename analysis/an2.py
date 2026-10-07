import json, math, statistics as st
D=json.load(open('e2_k.json')); R=D['rows']; KS=D['ks']
def lo(r,k,a,tag=''): return r['k%d_%s%s_lo'%(k,tag,a)]
def dd(r,k,tag=''):   return lo(r,k,'harv',tag)-lo(r,k,'self',tag)

def pear(x,y):
    n=len(x); mx=st.mean(x); my=st.mean(y)
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    if sx==0 or sy==0: return float('nan')
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)
def rank(v):
    o=sorted(range(len(v)),key=lambda i:v[i]); rk=[0.0]*len(v); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
        r=(i+j)/2.0+1
        for t in range(i,j+1): rk[o[t]]=r
        i=j+1
    return rk
def spear(x,y): return pear(rank(x),rank(y))
def ci_r(r,n):
    if abs(r)>=1: return (r,r)
    z=0.5*math.log((1+r)/(1-r)); se=1/math.sqrt(n-3)
    f=lambda t:(math.exp(2*t)-1)/(math.exp(2*t)+1)
    return f(z-1.96*se), f(z+1.96*se)

n=len(R)
print('=== A · same item, different budgets k: is harv-self consistent? ===')
print('(the k=3 and k=5 serve sets are nested, so some correlation is expected; k=1 and k=5 share only 1 row)')
for a in KS:
    for b in KS:
        if a<b:
            x=[dd(r,a) for r in R]; y=[dd(r,b) for r in R]
            rp=pear(x,y); l,h=ci_r(rp,n)
            print('  k=%d vs k=%d :  pearson %+0.3f [%+0.3f,%+0.3f]   spearman %+0.3f'%(a,b,rp,l,h,spear(x,y)))

print('\n=== B · same item, real serve set vs placebo serve set (entirely different rows) ===')
print('(correlated => wins/losses are a property of the question, not of the content served)')
for k in KS:
    x=[dd(r,k) for r in R]; y=[dd(r,k,'p_') for r in R]
    rp=pear(x,y); l,h=ci_r(rp,n)
    print('  k=%d :  pearson %+0.3f [%+0.3f,%+0.3f]   spearman %+0.3f'%(k,rp,l,h,spear(x,y)))

print('\n=== C · reference: the same two tests on the self-iso channel, which has a real effect ===')
def ds(r,k,tag=''): return lo(r,k,'self',tag)-lo(r,k,'iso',tag)
for a,b in ((3,5),):
    x=[ds(r,a) for r in R]; y=[ds(r,b) for r in R]
    print('  self-iso  k=%d vs k=%d : pearson %+0.3f'%(a,b,pear(x,y)))
for k in (5,):
    x=[ds(r,k) for r in R]; y=[ds(r,k,'p_') for r in R]
    print('  self-iso  k=%d real vs placebo : pearson %+0.3f'%(k,pear(x,y)))

print('\n=== D · are wins/losses driven by how extreme the item is overall? ===')
for k in (5,):
    ax=[abs(dd(r,k)) for r in R]
    sc=[max(abs(lo(r,k,'iso')),abs(lo(r,k,'self')),abs(lo(r,k,'harv'))) for r in R]
    print('  |harv-self| vs item log-odds magnitude :  pearson %+0.3f  spearman %+0.3f'%(pear(ax,sc),spear(ax,sc)))
    nn=[abs(r['r_none_lo']) for r in R]
    print('  |harv-self| vs |no-context prior|      :  pearson %+0.3f  spearman %+0.3f'%(pear(ax,nn),spear(ax,nn)))
