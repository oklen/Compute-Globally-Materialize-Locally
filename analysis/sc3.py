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
def mk(nm,fn): return (nm,fn)
READ=[mk('judge (bit)',lambda r,A,B: float(r['j_'+A]-r['j_'+B])),
      mk('token-F1',  lambda r,A,B: r['f1_'+A]-r['f1_'+B]),
      mk('precision', lambda r,A,B: r['pr_'+A]-r['pr_'+B])]
for k in (1,2,3,5):
    READ.append(mk('forced-choice acc k=%d'%k,(lambda k:lambda r,A,B: None if not r['_kk'] else
        (1. if r['_kk']['k%d_%s_p'%(k,A)]>.5 else 0.)-(1. if r['_kk']['k%d_%s_p'%(k,B)]>.5 else 0.))(k)))
    READ.append(mk('log-odds k=%d'%k,(lambda k:lambda r,A,B: None if not r['_kk'] else
        r['_kk']['k%d_%s_lo'%(k,A)]-r['_kk']['k%d_%s_lo'%(k,B)])(k)))
def run(rows,fn,A='harv',B='self'):
    rs=[r for r in rows if fn(r,A,B) is not None]
    if len(rs)<25: return None
    d=[fn(r,A,B) for r in rs]; g=[IDX[r['conv']] for r in rs]
    if len({x for x in g})<3: return None
    w=W(d,g,G=len(CONV)); L,H=w.ci()
    return len(d),w.mean,w.p(),L,H
print('='*118)
print('A · global: harv vs self, each ruler + placebo control')
print('='*118)
print('%-15s %5s %+11s %7s %24s | %11s %7s %s'%('ruler','n','Δ','WCR p','95% CI','placeboΔ','p','verdict'))
NT=0; hits=[]
for nm,fn in READ:
    a=run(RO,fn); b=run(RO,fn,'p_harv','p_self')
    if not a: continue
    NT+=1
    n,m,p,L,H=a
    tag='harv wins' if (p<.05 and m>0) else ('self wins' if (p<.05 and m<0) else 'tie (n.s.)')
    if p<.05 and m>0 and b and b[2]>=.05: tag+=' ★placebo flat'
    if p<.05 and m>0: hits.append((nm,'global',n,m,p,b[1] if b else None,b[2] if b else None))
    print('%-15s %5d %+11.4f %7.4f  [%+9.4f,%+9.4f] | %+11.4f %7.4f %s'%(nm,n,m,p,L,H,b[1] if b else float('nan'),b[2] if b else float('nan'),tag))
print()
print('='*118)
print('B · subgroup search: each ruler × each subgroup (only harv significantly positive listed; total test count at the end)')
print('='*118)
NTsub=0
def tert(vals):
    v=sorted(vals); return v[len(v)//3], v[2*len(v)//3]
t1,t2=tert([r['_nt'] for r in RO]); d1,d2=tert([r['_depth'] for r in RO])
SUB=[('multi-hop cat1',lambda r:r['cat']==1),('temporal cat2',lambda r:r['cat']==2),
     ('open cat3',lambda r:r['cat']==3),('single-hop cat4',lambda r:r['cat']==4),
     ('shortest conv 1/3',lambda r:r['_nt']<=t1),('longest conv 1/3',lambda r:r['_nt']>t2),
     ('earliest served rows 1/3',lambda r:r['_depth']<=d1),('latest served rows 1/3',lambda r:r['_depth']>d2),
     ('serve set has gold evidence',lambda r:bool(r['serve_has_goldev'])),
     ('serve set lacks gold evidence',lambda r:not r['serve_has_goldev'])]
rowsout=[]
for nm,fn in READ:
    for sn,sf in SUB:
        sub=[r for r in RO if sf(r)]
        a=run(sub,fn)
        if not a: continue
        NTsub+=1
        n,m,p,L,H=a
        if p<.05 and m>0:
            b=run(sub,fn,'p_harv','p_self')
            rowsout.append((nm,sn,n,m,p,b[1] if b else None,b[2] if b else None))
print('%-15s %-20s %5s %+11s %7s %11s %7s'%('ruler','subgroup','n','Δ','WCR p','placeboΔ','p'))
for nm,sn,n,m,p,bm,bp in rowsout:
    print('%-15s %-20s %5d %+11.4f %7.4f %+11.4f %7.4f'%(nm,sn,n,m,p,bm if bm is not None else float('nan'),bp if bp is not None else float('nan')))
if not rowsout: print('  (none)')
print()
print('total tests: global %d + subgroup %d = %d. Subgroup hits: %d.'%(NT,NTsub,NT+NTsub,len(rowsout)))
print('at a nominal .05 level, %d tests expect about %.1f false positives.'%(NT+NTsub,0.05*(NT+NTsub)))
