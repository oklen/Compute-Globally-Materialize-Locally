import math, statistics as st, itertools
def make(G):
    return list(itertools.product([-1,1],repeat=G))
class W:
    """WCR (null imposed, Rademacher, full enumeration) -- pre-aggregated by cluster, O(G) per sign vector"""
    def __init__(self, d, g, G=None):
        self.G = G or (max(g)+1)
        self.n = len(d)
        self.ng=[0]*self.G; self.sd=[0.0]*self.G
        for x,gg in zip(d,g): self.ng[gg]+=1; self.sd[gg]+=x
        self.mean = sum(self.sd)/self.n
        self.S = make(self.G)
    def _t(self, Sg):
        n=self.n; mm=sum(Sg)/n
        s=0.0
        for gg in range(self.G):
            z=Sg[gg]-self.ng[gg]*mm; s+=z*z
        return (mm/(math.sqrt(s)/n)) if s>0 else 0.0
    def p(self, mu0=0.0):
        Sg=[self.sd[gg]-self.ng[gg]*mu0 for gg in range(self.G)]
        t=self._t(Sg); c=0
        for w in self.S:
            if abs(self._t([w[gg]*Sg[gg] for gg in range(self.G)]))>=abs(t)-1e-12: c+=1
        return c/len(self.S)
    def ci(self, alpha=0.05, steps=34, half=None):
        m=self.mean
        if half is None:
            v=math.sqrt(sum((self.sd[gg]-self.ng[gg]*m/self.G*0)**2 for gg in range(self.G)))/self.n
            half=max(abs(m)*8, v*8, 1e-6)
        def rej(mu): return self.p(mu)<alpha
        lo=m-half
        while not rej(lo) and half<1e9: half*=2; lo=m-half
        hi=m+half
        while not rej(hi) and half<1e9: half*=2; hi=m+half
        a,b=lo,m
        for _ in range(steps):
            mid=(a+b)/2
            if rej(mid): a=mid
            else: b=mid
        L=(a+b)/2
        a,b=m,hi
        for _ in range(steps):
            mid=(a+b)/2
            if rej(mid): b=mid
            else: a=mid
        return L,(a+b)/2
