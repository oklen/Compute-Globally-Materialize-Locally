import json, re, glob, statistics as st
RO=json.load(open('e2_ro.json'))['rows']
pat=re.compile(r'^\[e2\] (\S+) q(\d+) .*\| LO ([^|]+)\|')
LO={}
for f in sorted(glob.glob('ro_sh?.log')):
    for line in open(f):
        m=pat.match(line)
        if not m: continue
        v=m.group(3).split()
        if any(x=='nan' for x in v): continue
        LO[(m.group(1),int(m.group(2)))]=[float(x) for x in v]  # none,iso,self,harv,p_iso,p_self,p_harv
SL={'none':0,'iso':1,'self':2,'harv':3}
def norm(s): return re.sub(r'\s+',' ',(s or '').strip().lower().rstrip('.'))
sub=[r for r in RO if (str(r['conv']),r['qi']) in LO]
print('subset with a log-odds readout n=%d\n'%len(sub))
print('%-12s %6s %8s %8s %8s %8s %10s %14s %12s'%('contrast','all n','both right','both wrong','A right B wrong','B right A wrong','diff=0 share','of which same text','Δlo on ties'))
for A,B in (('self','iso'),('harv','iso'),('harv','self')):
    for tag,rows in (('599 subset',sub),('all 1269',RO)):
        c={}
        for r in rows: c[(r['j_'+A],r['j_'+B])]=c.get((r['j_'+A],r['j_'+B]),0)+1
        tie=c.get((1,1),0)+c.get((0,0),0); n=len(rows)
        tr=[r for r in rows if r['j_'+A]==r['j_'+B]]
        same=sum(1 for r in tr if norm(r['ans_'+A])==norm(r['ans_'+B]))
        if tag=='599 subset':
            dl=[LO[(str(r['conv']),r['qi'])][SL[A]]-LO[(str(r['conv']),r['qi'])][SL[B]] for r in tr]
            dls='%+0.4f'%st.mean(dl)
        else: dls='  -'
        print('%-12s %6d %8d %8d %8d %8d %9.1f%% %13.1f%% %12s'
              %('%s-%s %s'%(A,B,tag),n,c.get((1,1),0),c.get((0,0),0),c.get((1,0),0),c.get((0,1),0),
                100*tie/n,100*same/len(tr),dls))
print()
print('=== the root cause of ties: both arms are wrong ===')
for A,B in (('self','iso'),('harv','iso')):
    c={}
    for r in sub: c[(r['j_'+A],r['j_'+B])]=c.get((r['j_'+A],r['j_'+B]),0)+1
    tie=c.get((1,1),0)+c.get((0,0),0)
    print('  %s-%s : of %d ties, %d (%.0f%%) have both arms wrong'%(A,B,tie,c.get((0,0),0),100*c.get((0,0),0)/tie))
print('\n  per-arm judge accuracy (599 subset):', {a:round(sum(r['j_'+a] for r in sub)/len(sub),3) for a in ('none','text','iso','self','harv')})
print('  per-arm judge accuracy (all 1269):', {a:round(sum(r['j_'+a] for r in RO)/len(RO),3) for a in ('none','text','iso','self','harv')})
