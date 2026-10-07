import json, re, math, statistics as st
K=json.load(open('e2_k.json')); Kr={(str(r['conv']),r['qi']):r for r in K['rows']}
pat=re.compile(r'^\[e2\] (\S+) q(\d+) .*\| LO ([^|]+)\| PG ([^|]+)\|')
RO={}
import glob
for f in sorted(glob.glob('ro_sh?.log')):
    for line in open(f):
        m=pat.match(line)
        if not m: continue
        lo=m.group(3).split(); pg=m.group(4).split()
        if any(v=='nan' for v in lo): continue
        RO[(m.group(1),int(m.group(2)))]=([float(v) for v in lo],[float(v) for v in pg])
print('e2_ro items with a usable readout: %d / 1269'%len(RO))
common=[k for k in Kr if k in RO]
print('intersection with e2_k (n=%d): %d\n'%(len(Kr),len(common)))

# locate the order of the 7 values in LO: anchor on none, then compare each with the k5 arms of e2_k
k0=common[0]
print('first fix the arm order of the 7 logged values (example item %s):'%str(k0))
print('  logged LO:', RO[k0][0])
r=Kr[k0]
print('  e2_k   : none %.4f | iso %.4f self %.4f harv %.4f | p_iso %.4f p_self %.4f p_harv %.4f'
      %(r['r_none_lo'],r['k5_iso_lo'],r['k5_self_lo'],r['k5_harv_lo'],r['k5_p_iso_lo'],r['k5_p_self_lo'],r['k5_p_harv_lo']))
# automatic matching: score every candidate permutation
cands={'none':'r_none_lo','iso':'k5_iso_lo','self':'k5_self_lo','harv':'k5_harv_lo',
       'p_iso':'k5_p_iso_lo','p_self':'k5_p_self_lo','p_harv':'k5_p_harv_lo'}
print('\nexact-equality rate of each logged slot against each e2_k column (%d items):'%len(common))
print('  %-8s'%'slot', ''.join('%9s'%n for n in cands))
for j in range(7):
    row=[]
    for n,f in cands.items():
        eq=sum(1 for k in common if abs(RO[k][0][j]-Kr[k][f])<1e-9)
        row.append(eq/len(common))
    print('  slot%-4d'%j, ''.join('%9.3f'%v for v in row))
