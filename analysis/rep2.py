import json, re, glob, statistics as st
K=json.load(open('e2_k.json')); Kr={(str(r['conv']),r['qi']):r for r in K['rows']}
pat=re.compile(r'^\[e2\] (\S+) q(\d+) .*\| LO ([^|]+)\| PG ([^|]+)\|')
RO={}
for f in sorted(glob.glob('ro_sh?.log')):
    for line in open(f):
        m=pat.match(line)
        if not m: continue
        lo=m.group(3).split(); pg=m.group(4).split()
        if any(v=='nan' for v in lo): continue
        RO[(m.group(1),int(m.group(2)))]=([float(v) for v in lo],[float(v) for v in pg])
ORD=[('none','r_none_lo','r_none_p'),('iso','k5_iso_lo','k5_iso_p'),('self','k5_self_lo','k5_self_p'),
     ('harv','k5_harv_lo','k5_harv_p'),('p_iso','k5_p_iso_lo','k5_p_iso_p'),
     ('p_self','k5_p_self_lo','k5_p_self_p'),('p_harv','k5_p_harv_lo','k5_p_harv_p')]
com=[k for k in Kr if k in RO]
nlo=nlo_d=npg=npg_d=0; mx_lo=mx_pg=0.0; bad=[]
for k in com:
    lo,pg=RO[k]; r=Kr[k]
    for j,(nm,fl,fp) in enumerate(ORD):
        d1=abs(lo[j]-r[fl]); d2=abs(pg[j]-r[fp])
        nlo+=1; npg+=1; mx_lo=max(mx_lo,d1); mx_pg=max(mx_pg,d2)
        if d1>0: nlo_d+=1; bad.append((k,nm,lo[j],r[fl]))
        if d2>0: npg_d+=1
print('two independent runs (e2_ro, 4 shards, 04:54  vs  e2_k 05:53) compared per item and arm')
print('  items %d × arms %d = %d number pairs'%(len(com),len(ORD),nlo))
print('  log-odds  differing pairs: %d   max abs diff: %.3e'%(nlo_d,mx_lo))
print('  p(gold)   differing pairs: %d   max abs diff: %.3e'%(npg_d,mx_pg))
if bad: print('  example disagreements:',bad[:5])
print()
print('  => verdict: %s'%('bit-for-bit identical' if nlo_d==0 and npg_d==0 else 'disagreements exist'))
