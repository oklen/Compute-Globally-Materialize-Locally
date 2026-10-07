import json, re, glob, string, math, statistics as st, itertools, collections
RO=json.load(open('e2_ro.json'))['rows']
def norm(s):
    s=(s or '').lower(); s=''.join(c for c in s if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def prf(pred,gold):
    p=norm(pred).split(); g=norm(gold).split()
    if not p or not g: return (float(p==g),)*3
    c=collections.Counter(p)&collections.Counter(g); ns=sum(c.values())
    if ns==0: return (0.,0.,0.)
    pr=ns/len(p); rc=ns/len(g); return (2*pr*rc/(pr+rc),pr,rc)
ARMS=('none','text','iso','self','harv','p_iso','p_self','p_harv')
for r in RO:
    for a in ARMS:
        f,p_,r_=prf(r.get('ans_'+a,''),r['gold'])
        r['f1_'+a]=f; r['pr_'+a]=p_; r['rc_'+a]=r_
        r['len_'+a]=len(norm(r.get('ans_'+a,'')).split())
CONV=sorted({r['conv'] for r in RO},key=lambda x:int(x)); IDX={c:i for i,c in enumerate(CONV)}
SIGNS=list(itertools.product([-1,1],repeat=len(CONV)))
def wcr(d,g):
    n=len(d); m=st.mean(d)
    def T(v):
        mm=st.mean(v); byg={}
        for x,gg in zip(v,g): byg.setdefault(gg,[]).append(x-mm)
        s=sum((sum(z))**2 for z in byg.values())
        return mm/(math.sqrt(s)/n) if s>0 else 0.0
    t=T(d); return m,sum(1 for w in SIGNS if abs(T([w[gg]*x for x,gg in zip(d,g)]))>=abs(t)-1e-12)/len(SIGNS)
g=[IDX[r['conv']] for r in RO]
print('=== is the positive F1 bought by longer answers? (n=%d) ==='%len(RO))
print('  %-8s %10s %10s %10s %10s'%('arm','answer words','precision','recall','F1'))
for a in ('iso','self','harv','p_iso','p_self','p_harv'):
    print('  %-8s %10.2f %10.4f %10.4f %10.4f'%(a,st.mean([r['len_'+a] for r in RO]),
          st.mean([r['pr_'+a] for r in RO]),st.mean([r['rc_'+a] for r in RO]),st.mean([r['f1_'+a] for r in RO])))
print()
for A,B in (('harv','self'),('harv','iso'),('self','iso'),('p_harv','p_self')):
    for m,nm in (('len','words'),('pr','precision'),('rc','recall'),('f1','F1')):
        d=[r[m+'_'+A]-r[m+'_'+B] for r in RO]; v,p=wcr(d,g)
        print('  %-12s Δ%-10s %+8.4f  (WCR p %.4f)'%('%s-%s'%(A,B) if m=='len' else '',nm,v,p))
    print()
