import json, math, statistics as st, random
D=json.load(open('e2_k.json')); R=D['rows']
def P(r,k,a,t=''): return r['k%d_%s%s_p'%(k,t,a)]
def L(r,k,a,t=''): return r['k%d_%s%s_lo'%(k,t,a)]
def pear(x,y):
    mx=st.mean(x);my=st.mean(y);sx=math.sqrt(sum((a-mx)**2 for a in x));sy=math.sqrt(sum((b-my)**2 for b in y))
    return float('nan') if sx*sy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)

print('=== is the noise independent of self? (independence is the premise of the additive-noise model) ===')
for k in (2,3,5):
    s=[L(r,k,'self') for r in R]; e=[L(r,k,'harv')-L(r,k,'self') for r in R]
    print('   k=%d  corr(self, harv-self) = %+0.3f'%(k,pear(s,e)))

print('\n=== decisive test: randomly reassign the harv-self differences across items ===')
print('   If harv is just self plus content-free jitter, the shuffled accuracy should reproduce the real harv accuracy.')
print('   (the marginal distribution of the jitter is kept exactly; only the link between jitter and item is cut)\n')
random.seed(1234); B=2000
print('   %-6s %-8s %10s %12s %14s %10s'%('k','serve set','self acc','harv observed','shuffled pred.','observed pctile in pred.'))
for k in (2,3,5):
    for tag,nm in (('','real'),('p_','placebo')):
        s=[L(r,k,'self',tag) for r in R]; e=[L(r,k,'harv',tag)-L(r,k,'self',tag) for r in R]
        gold=[1 if r['gold_is_a'] else 1 for r in R]
        obs=sum(1 for r in R if P(r,k,'harv',tag)>.5)/len(R)
        selfacc=sum(1 for r in R if P(r,k,'self',tag)>.5)/len(R)
        sims=[]
        for _ in range(B):
            ee=e[:]; random.shuffle(ee)
            sims.append(sum(1 for a,b in zip(s,ee) if a+b>0)/len(s))
        sims.sort(); q=sum(1 for v in sims if v<obs)/B
        print('   k=%-4d %-8s %10.4f %12.4f %14.4f %10.3f  [%.4f,%.4f]'
              %(k,nm,selfacc,obs,st.mean(sims),q,sims[int(.025*B)],sims[int(.975*B)]))

print('\n=== same shuffle: can the log-odds-layer Δ accuracy (harv-self) be reproduced? ===')
for k in (5,):
    for tag,nm in (('','real'),('p_','placebo')):
        s=[L(r,k,'self',tag) for r in R]; e=[L(r,k,'harv',tag)-L(r,k,'self',tag) for r in R]
        selfacc=sum(1 for a in s if a>0)/len(s)
        obsd=sum(1 for r in R if P(r,k,'harv',tag)>.5)/len(R)-sum(1 for r in R if P(r,k,'self',tag)>.5)/len(R)
        sims=[]
        for _ in range(B):
            ee=e[:]; random.shuffle(ee)
            sims.append(sum(1 for a,b in zip(s,ee) if a+b>0)/len(s)-selfacc)
        sims.sort()
        print('   k=%d %-6s  observed Δacc %+0.4f   shuffled %+0.4f  95%% [%+0.4f,%+0.4f]  -> %s'
              %(k,nm,obsd,st.mean(sims),sims[int(.025*B)],sims[int(.975*B)],
                'inside the interval (indistinguishable from random jitter)' if sims[int(.025*B)]<=obsd<=sims[int(.975*B)] else 'outside the interval'))
