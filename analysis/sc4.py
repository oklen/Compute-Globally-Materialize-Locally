import json,re,string,math,statistics as st,collections,sys
sys.path.insert(0,'.'); from wcrfast import W
RO=json.load(open('e2_ro.json'))['rows']
KM={(str(r['conv']),r['qi']):r for r in json.load(open('e2_k.json'))['rows']}
S={(str(i['conv']),i['qi']):i for i in json.load(open('e2_sets.json'))['items']}
def norm(s):
    s=(s or '').lower(); s=''.join(c for c in s if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def prf(p,gld):
    p=norm(p).split(); g=norm(gld).split()
    if not p or not g: return (float(p==g),)*2
    c=collections.Counter(p)&collections.Counter(g); ns=sum(c.values())
    if ns==0: return (0.,0.)
    pr=ns/len(p); rc=ns/len(g); return (2*pr*rc/(pr+rc),pr)
for r in RO:
    for a in ('iso','self','harv','p_iso','p_self','p_harv'):
        f,p_=prf(r.get('ans_'+a,''),r['gold']); r['f1_'+a]=f; r['pr_'+a]=p_
    s=S[(str(r['conv']),r['qi'])]; r['_nt']=s['n_turns']; r['_depth']=min(r['serve'])/s['n_turns']
    r['_kk']=KM.get((str(r['conv']),r['qi']))
CONV=sorted({r['conv'] for r in RO},key=lambda x:int(x)); IDX={c:i for i,c in enumerate(CONV)}
def tert(v): v=sorted(v); return v[len(v)//3],v[2*len(v)//3]
t1,t2=tert([r['_nt'] for r in RO]); d1,d2=tert([r['_depth'] for r in RO])
SUB=[('all',lambda r:True),('multi-hop cat1',lambda r:r['cat']==1),('temporal cat2',lambda r:r['cat']==2),
     ('open cat3',lambda r:r['cat']==3),('single-hop cat4',lambda r:r['cat']==4),
     ('shortest conv 1/3',lambda r:r['_nt']<=t1),('longest conv 1/3',lambda r:r['_nt']>t2),
     ('earliest served 1/3',lambda r:r['_depth']<=d1),('latest served 1/3',lambda r:r['_depth']>d2),
     ('serve set has gold evidence',lambda r:bool(r['serve_has_goldev'])),('serve set lacks gold evidence',lambda r:not r['serve_has_goldev'])]
READ=[('judge',lambda r,A,B: float(r['j_'+A]-r['j_'+B])),
      ('token-F1',lambda r,A,B: r['f1_'+A]-r['f1_'+B]),
      ('precision',lambda r,A,B: r['pr_'+A]-r['pr_'+B]),
      ('forced-choice acc k=5',lambda r,A,B: None if not r['_kk'] else (1. if r['_kk']['k5_%s_p'%A]>.5 else 0.)-(1. if r['_kk']['k5_%s_p'%B]>.5 else 0.))]
def run(rows,fn,A='harv',B='self'):
    rs=[r for r in rows if fn(r,A,B) is not None]
    if len(rs)<25 or len({r['conv'] for r in rs})<3: return None
    d=[fn(r,A,B) for r in rs]; g=[IDX[r['conv']] for r in rs]
    w=W(d,g,G=len(CONV)); return len(d),w.mean,w.p()
for nm,fn in READ:
    print('\n=== %s: harv - self, all subgroups (with the same contrast on placebo) ==='%nm)
    print('  %-18s %6s %+11s %8s | %11s %8s'%('subgroup','n','Δ','WCR p','placeboΔ','p'))
    for sn,sf in SUB:
        sub=[r for r in RO if sf(r)]
        a=run(sub,fn); b=run(sub,fn,'p_harv','p_self')
        if not a: print('  %-18s %6d  (too few items or clusters, skipped)'%(sn,len(sub))); continue
        mk=''
        if a[2]<.05 and a[1]>0: mk=' ★' if (b and b[2]>=.05 and abs(b[1])<abs(a[1])/2) else ' ✗placebo moves too'
        if a[2]<.05 and a[1]<0: mk=' <- self wins'
        print('  %-18s %6d %+11.4f %8.4f | %+11.4f %8.4f%s'%(sn,a[0],a[1],a[2],
              b[1] if b else float('nan'),b[2] if b else float('nan'),mk))
