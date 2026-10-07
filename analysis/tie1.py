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
        LO[(m.group(1),int(m.group(2)))]=[float(x) for x in v]   # none,iso,self,harv,p_iso,p_self,p_harv
def norm(s): return re.sub(r'\s+',' ',(s or '').strip().lower().rstrip('.'))

print('=== per-item harv vs iso difference on the judge ruler (j in {0,1}) n=%d ==='%len(RO))
cell={}
for r in RO:
    cell[(r['j_harv'],r['j_iso'])]=cell.get((r['j_harv'],r['j_iso']),0)+1
n=len(RO)
tie=cell.get((1,1),0)+cell.get((0,0),0)
print('  both arms right : %4d'%cell.get((1,1),0))
print('  both arms wrong : %4d   <-- the vast majority'%cell.get((0,0),0))
print('  harv right/iso wrong: %4d'%cell.get((1,0),0))
print('  iso right/harv wrong: %4d'%cell.get((0,1),0))
print('  --> items with a difference of exactly 0: %d / %d = %.1f%%'%(tie,n,100*tie/n))

print('\n=== are these zeros identical answers, or a judge that cannot see a difference? ===')
same_txt=sum(1 for r in RO if norm(r['ans_harv'])==norm(r['ans_iso']))
tie_rows=[r for r in RO if r['j_harv']==r['j_iso']]
tie_same=sum(1 for r in tie_rows if norm(r['ans_harv'])==norm(r['ans_iso']))
print('  of all %d items, the two arms generated identical answers: %d (%.1f%%)'%(n,same_txt,100*same_txt/n))
print('  of the %d items with diff=0, identical answers: %d (%.1f%%)'%(len(tie_rows),tie_same,100*tie_same/len(tie_rows)))
print('  => of diff=0, %d items (%.1f%%) are [different answers, same judge verdict]'
      %(len(tie_rows)-tie_same,100*(len(tie_rows)-tie_same)/len(tie_rows)))

print('\n=== on items the judge scores 0, how far apart are the internal states? (log-odds, harv-iso) ===')
def lodiff(r):
    k=(str(r['conv']),r['qi'])
    return None if k not in LO else LO[k][3]-LO[k][1]
for lab,sel in (('both wrong (0,0)',lambda r:r['j_harv']==0 and r['j_iso']==0),
                ('both right (1,1)',lambda r:r['j_harv']==1 and r['j_iso']==1),
                ('harv right/iso wrong', lambda r:r['j_harv']==1 and r['j_iso']==0),
                ('iso right/harv wrong', lambda r:r['j_harv']==0 and r['j_iso']==1)):
    v=[lodiff(r) for r in RO if sel(r)]; v=[x for x in v if x is not None]
    if not v: print('  %-16s (no log-odds readout)'%lab); continue
    nz=sum(1 for x in v if abs(x)>1e-9)
    print('  %-16s n=%3d  mean %+7.4f  median %+7.4f  share |Δ|>0 %.1f%%  max|Δ| %.2f'
          %(lab,len(v),st.mean(v),st.median(v),100*nz/len(v),max(abs(x) for x in v)))

print('\n=== contrast: on the log-odds ruler, how many are exactly 0? ===')
allv=[LO[k][3]-LO[k][1] for k in LO]
print('  n=%d  exactly 0: %d (%.1f%%)'%(len(allv),sum(1 for x in allv if abs(x)<1e-9),
      100*sum(1 for x in allv if abs(x)<1e-9)/len(allv)))
