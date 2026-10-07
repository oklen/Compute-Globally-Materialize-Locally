"""Follow-up analyses for Q5 and Q2b: demonstrate greedy underestimation directly, and turn the attn_traj asymmetry into a same-item pairing."""
import json, math, random, statistics as st, sys
P = sys.argv[1]
R = json.load(open(P))["rows"]
BUDS = (5, 10, 20, 40)
B = 4000

def boot(xs, f=st.fmean, seed=1234):
    if len(xs) < 4: return (None, None)
    rnd = random.Random(seed); n = len(xs)
    o = sorted(f([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return o[int(.025*B)], o[int(.975*B)]
def star(xs):
    lo, hi = boot(xs)
    return "*" if lo is not None and lo*hi > 0 else " "

print("="*90)
print("Q5-a  direct demonstration of greedy underestimation: keep the top B% by oracle density, sum the evict of the dropped units as the 'predicted loss',")
print("      and compare with the actual loss lo(full) - lo(B). Ratio >1 = the actual loss hurts more than the per-unit prediction.")
for KIND in ("unstated", "stated"):
    live = [r for r in R if r.get(KIND+"_full") is not None and abs(r.get(KIND+"_none",0))<3
            and abs(r[KIND+"_full"]-r[KIND+"_frame"])>2]
    if not live: continue
    print("\n  --- %s  n=%d ---" % (KIND, len(live)))
    for g in ("passage","sentence","block"):
        row=[]
        for b in BUDS:
            k="ret_%s_oracle_%d"%(KIND,b); rats=[]
            for r in live:
                u=r.get("%s_%s"%(KIND,g))
                if not u or k not in r: continue
                tot=sum(x["n"] for x in u); cap=int(round(b/100.0*tot))
                o=sorted(u,key=lambda x:-x["dens"]); n=0; kept=set()
                for i,x in enumerate(o):
                    if n>=cap: break
                    kept.add(i); n+=x["n"]
                pred=sum(x["evict"] for i,x in enumerate(o) if i not in kept)
                act=r[KIND+"_full"]-r[k]
                if pred>1.0: rats.append(act/pred)
            row.append("%.2f%s"%(st.median(rats), star(rats)) if len(rats)>=4 else " na ")
        print("    %-9s actual/predicted:  %s   (n=%d)"%(g,"  ".join("B=%d%% %s"%(b,c) for b,c in zip(BUDS,row)),len(rats)))

print("\n"+"="*90)
print("Q5-b  mechanism: separate items whose value recurs from items where it appears once. If redundancy is the cause, recurring items should show a lower ratio.")
for KIND in ("unstated","stated"):
    live=[r for r in R if r.get(KIND+"_full") is not None and abs(r.get(KIND+"_none",0))<3
          and abs(r[KIND+"_full"]-r[KIND+"_frame"])>2 and r.get(KIND+"_sentence")]
    if not live: continue
    print("\n  --- %s ---"%KIND)
    for g in ("passage","sentence","block"):
        grp={0:[],1:[]}
        for r in live:
            u=r["%s_%s"%(KIND,g)] if ("%s_%s"%(KIND,g)) in r else None
            if not u: continue
            nhit=sum(1 for x in r[KIND+"_sentence"] if x["has_gold"])
            w=r[KIND+"_full"]-r[KIND+"_frame"]
            if abs(w)<=2: continue
            grp[1 if nhit>=2 else 0].append(sum(x["evict"] for x in u)/w)
        if len(grp[0])>=4 and len(grp[1])>=4:
            print("    %-9s value in 1 sentence: %.2f (n=%d)   value in >=2 sentences: %.2f (n=%d)"
                  %(g,st.median(grp[0]),len(grp[0]),st.median(grp[1]),len(grp[1])))

print("\n"+"="*90)
print("Q2b  same-item pairing: same trajectories, same ranker, only the follow-up target changes.")
both=[r for r in R if r.get("stated_full") is not None and r.get("unstated_full") is not None
      and abs(r.get("stated_none",0))<3 and abs(r.get("unstated_none",0))<3
      and abs(r["stated_full"]-r["stated_frame"])>2 and abs(r["unstated_full"]-r["unstated_frame"])>2]
print("  items with a valid denominator in both layers: n=%d"%len(both))
for rk in ("attn_traj","attn_h2o","attn_snap","bm25_now","recency"):
    cells=[]
    for b in BUDS:
        s=[(r["ret_stated_%s_%d"%(rk,b)]-r["ret_stated_random_%d"%(b)])/(r["stated_full"]-r["stated_frame"]) for r in both]
        u=[(r["ret_unstated_%s_%d"%(rk,b)]-r["ret_unstated_random_%d"%(b)])/(r["unstated_full"]-r["unstated_frame"]) for r in both]
        d=[a-c for a,c in zip(s,u)]
        cells.append("stated %+.3f%s / unstated %+.3f%s / diff %+.3f%s"%(st.fmean(s),star(s),st.fmean(u),star(u),st.fmean(d),star(d)))
    print("  %-10s"%rk)
    for b,c in zip(BUDS,cells): print("      B=%2d%%  %s"%(b,c))

# ---- Q5-c how redundancy inflates raw recovery (every ranker gets it, random included) ----
print("\n" + "=" * 90)
print("Q5-c  redundancy raises the raw recovery of **every** ranker -- so raw recovery is not comparable across corpora")
K = "unstated"
live = [r for r in R if r.get(K+"_full") is not None and abs(r.get(K+"_none",0)) < 3
        and abs(r[K+"_full"]-r[K+"_frame"]) > 2 and r.get(K+"_sentence")]
g1 = [r for r in live if sum(1 for x in r[K+"_sentence"] if x["has_gold"]) < 2]
g2 = [r for r in live if sum(1 for x in r[K+"_sentence"] if x["has_gold"]) >= 2]
print("  value in 1 sentence n=%d   value in >=2 sentences n=%d  (%.0f%% redundant)" % (len(g1), len(g2), 100*len(g2)/len(live)))
print("  %-11s %-6s %-14s %-14s %s" % ("ranker", "budget", "value in 1", "value in >=2", "diff"))
for rk in ("oracle", "attn_h2o", "attn_snap", "attn_traj", "recency", "random"):
    for b in (10, 20):
        f = lambda G: [(r["ret_%s_%s_%d" % (K, rk, b)] - r[K+"_frame"]) / (r[K+"_full"] - r[K+"_frame"]) for r in G]
        a, c = f(g1), f(g2)
        print("  %-11s B=%2d%%  %.3f          %.3f          %+.3f"
              % (rk, b, st.fmean(a), st.fmean(c), st.fmean(a) - st.fmean(c)))
