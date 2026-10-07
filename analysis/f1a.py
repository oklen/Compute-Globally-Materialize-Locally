import json, re, glob, string, math, statistics as st, itertools, collections
RO=json.load(open('e2_ro.json'))['rows']
pat=re.compile(r'^\[e2\] (\S+) q(\d+) .*\| LO ([^|]+)\|')
LO={}
for f in sorted(glob.glob('ro_sh?.log')):
    for line in open(f):
        m=pat.match(line)
        if not m: continue
        v=m.group(3).split()
        if any(x=='nan' for x in v): continue
        LO[(m.group(1),int(m.group(2)))]=[float(x) for x in v]
SL={'none':0,'iso':1,'self':2,'harv':3}
# SQuAD normalize_answer
def norm(s):
    s=(s or '').lower()
    s=''.join(ch for ch in s if ch not in set(string.punctuation))
    s=re.sub(r'\b(a|an|the)\b',' ',s)
    return ' '.join(s.split())
def f1(pred,gold):
    p=norm(pred).split(); g=norm(gold).split()
    if not p or not g: return float(p==g)
    com=collections.Counter(p)&collections.Counter(g); ns=sum(com.values())
    if ns==0: return 0.0
    pr=ns/len(p); rc=ns/len(g); return 2*pr*rc/(pr+rc)
ARMS=('none','text','iso','self','harv','p_iso','p_self','p_harv')
for r in RO:
    for a in ARMS: r['f1_'+a]=f1(r.get('ans_'+a,''), r['gold'])

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

print('=== per-arm levels (n=1269, SQuAD token-F1 of the free-generated answer against gold) ===')
print('  %-8s %10s %10s'%('arm','judge accuracy','token-F1'))
for a in ARMS:
    print('  %-8s %10.4f %10.4f'%(a,sum(r['j_'+a] for r in RO)/len(RO),st.mean([r['f1_'+a] for r in RO])))

print('\n=== resolution: share of per-item differences that are exactly 0 ===')
print('  %-14s %14s %14s %14s'%('contrast','judge (bit)','token-F1','log-odds(599)'))
sub=[r for r in RO if (str(r['conv']),r['qi']) in LO]
for A,B in (('self','iso'),('harv','iso'),('harv','self')):
    zj=sum(1 for r in RO if r['j_'+A]==r['j_'+B])/len(RO)
    zf=sum(1 for r in RO if abs(r['f1_'+A]-r['f1_'+B])<1e-9)/len(RO)
    zl=sum(1 for r in sub if abs(LO[(str(r['conv']),r['qi'])][SL[A]]-LO[(str(r['conv']),r['qi'])][SL[B]])<1e-9)/len(sub)
    print('  %-14s %13.1f%% %13.1f%% %13.1f%%'%('%s-%s'%(A,B),100*zj,100*zf,100*zl))

print('\n=== three contrasts on token-F1 (n=1269, WCR, cluster = conversation) ===')
g=[IDX[r['conv']] for r in RO]
print('  %-14s %12s %10s | %s'%('contrast','Δ token-F1','WCR p','(ref.) judge Δ / WCR p'))
for A,B in (('self','iso'),('harv','iso'),('harv','self'),('p_self','p_iso'),('p_harv','p_iso'),('p_harv','p_self')):
    d=[r['f1_'+A]-r['f1_'+B] for r in RO]; m,p=wcr(d,g)
    dj=[float(r['j_'+A]-r['j_'+B]) for r in RO]; mj,pj=wcr(dj,g)
    print('  %-14s %+12.4f %10.4f | %+8.4f / %.4f'%('%s-%s'%(A,B),m,p,mj,pj))

print('\n=== the same three contrasts on the 599-item log-odds subset, three rulers side by side ===')
g2=[IDX[r['conv']] for r in sub]
print('  %-14s %14s %14s %16s'%('contrast','Δjudge','Δ token-F1','Δ log-odds'))
for A,B in (('self','iso'),('harv','iso'),('harv','self')):
    dj=[float(r['j_'+A]-r['j_'+B]) for r in sub]; mj,pj=wcr(dj,g2)
    df=[r['f1_'+A]-r['f1_'+B] for r in sub]; mf,pf=wcr(df,g2)
    dl=[LO[(str(r['conv']),r['qi'])][SL[A]]-LO[(str(r['conv']),r['qi'])][SL[B]] for r in sub]; ml,pl=wcr(dl,g2)
    print('  %-14s %+8.4f(%.3f) %+8.4f(%.3f) %+10.3f(%.4f)'%('%s-%s'%(A,B),mj,pj,mf,pf,ml,pl))
