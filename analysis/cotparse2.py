import re, json, math, statistics as st, sys, glob
K=re.compile(r'^\[e2k\] (\S+) q(\d+) \| k1 iso=(\S+) self=(\S+) harv=(\S+) p_iso=(\S+) p_self=(\S+) p_harv=(\S+) k5 iso=(\S+) self=(\S+) harv=(\S+) p_iso=(\S+) p_self=(\S+) p_harv=(\S+) \|')
C=re.compile(r'^\[e2cot\] (\S+) q(\d+) \| iso=(\S+) self=(\S+) harv=(\S+) p_iso=(\S+) p_self=(\S+) p_harv=(\S+) \| none=(\S+) mass=(\S+) \| mass=(\S+) ntok=(\d+) cap=(\d+)')
nc, cc = {}, {}
files=sorted(glob.glob('cot_*.log'))
for f in files:
    for line in open(f, errors='ignore'):
        m=K.match(line)
        if m:
            g=m.groups(); nc[(g[0],int(g[1]))]={
              'k1':dict(zip(('iso','self','harv','p_iso','p_self','p_harv'),map(float,g[2:8]))),
              'k5':dict(zip(('iso','self','harv','p_iso','p_self','p_harv'),map(float,g[8:14])))}
        m=C.match(line)
        if m:
            g=m.groups(); cc[(g[0],int(g[1]))]={
              'c5':dict(zip(('iso','self','harv','p_iso','p_self','p_harv'),map(float,g[2:8]))),
              'none':float(g[8]),'none_mass':float(g[9]),'harv_mass':float(g[10]),
              'ntok':int(g[11]),'cap':int(g[12])}
KS=sorted(set(nc)&set(cc))
R=[{'conv':k[0],'qi':k[1],
    **{'n_'+a:nc[k]['k5'][a] for a in nc[k]['k5']},
    **{'n1_'+a:nc[k]['k1'][a] for a in nc[k]['k1']},
    **{'c_'+a:cc[k]['c5'][a] for a in cc[k]['c5']},
    'c_none':cc[k]['none'],'none_mass':cc[k]['none_mass'],'harv_mass':cc[k]['harv_mass'],
    'ntok':cc[k]['ntok'],'cap':cc[k]['cap']} for k in KS]
json.dump(R, open('cot_all.json','w'))
print('logs %s'%files)
print('recovered %d items, %d conversation clusters: %s'%(len(R),len({r['conv'] for r in R}),sorted({r['conv'] for r in R})))
import collections
print('per cluster: %s'%dict(sorted(collections.Counter(r['conv'] for r in R).items())))
# k=1 identity self-check (must be 0 by construction)
z=max(abs(r['n1_self']-r['n1_iso']) for r in R)
print('k=1 null-by-construction self-check: max|self-iso| = %.8f  (must be 0)'%z)
print('CoT length %.1f  truncated %.1f%%  min mass %.4f  floor p_gold %.4f  saturated(|lo|>=15) %.1f%%'
      %(st.fmean([r['ntok'] for r in R]),100*st.fmean([r['cap'] for r in R]),
        min(r['harv_mass'] for r in R),
        st.fmean([1/(1+math.exp(-max(min(r['c_none'],60),-60))) for r in R]),
        100*st.fmean([abs(r['c_'+a])>=15 for r in R for a in ('iso','self','harv')])))
