"""Per-context concentration of the pool-referenced attribution, as predicted by P5."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
FIGDIR = _os.path.join(_HERE, "..", "figures") + _os.sep
import itertools, numpy as np, pandas as pd
from math import factorial
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED

from math import factorial
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
CFG={"RLV":RLV_CONFIG,"Healthcare":HEALTHCARE_CONFIG}
QMIN, TAU, STS = 0.50, 0.90, 8.0
NBG=256
RNG=np.random.default_rng(RNG_SEED+13)

def load(dom):
    df=pd.read_csv(D+"scenarios.csv"); d=df[df.domain==dom]
    a=d[d.stage_t==1].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    b=d[d.stage_t==2].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    g1=a[["g_S","g_A","g_D","g_E"]].values; g=b[["g_S","g_A","g_D","g_E"]].values
    q=b[["q_S","q_A","q_D","q_E"]].values; qn=q/q.sum(axis=1,keepdims=True)
    ts=np.exp(-((g-g1)*(STS*qn)*(g-g1)).sum(axis=1))
    return b,g,q,ts

def shapley(g,b,bd,idx):
    subs=[s for r in range(5) for s in itertools.combinations(range(4),r)]
    pools={c:g[(b.context_label==c).values] for c in bd}
    S=np.zeros((len(idx),4))
    for n,i in enumerate(idx):
        ctx=b.context_label[i]; smp=pools[ctx][RNG.integers(0,len(pools[ctx]),NBG)]
        v={}
        for s in subs:
            X=smp.copy()
            for k in s: X[:,k]=g[i,k]
            ok=np.ones(len(X),bool)
            for k,gn in enumerate(bd[ctx]):
                lo,hi=bd[ctx][gn]; ok&=(X[:,k]>=lo)&(X[:,k]<=hi)
            v[s]=ok.mean()
        for k in range(4):
            tot=0.0
            for s in subs:
                if k in s: continue
                w=factorial(len(s))*factorial(3-len(s))/factorial(4)
                tot+=w*(v[tuple(sorted(s+(k,)))]-v[s])
            S[n,k]=tot
    return S

rows=[]
for dom in ["RLV","Healthcare"]:
    cfg=CFG[dom]; b,g,q,ts=load(dom)
    bd={c:cfg.feasibility_bounds(c) for c in b.context_label.unique()}
    sl=np.zeros_like(g)
    for i,c in enumerate(b.context_label):
        for k,gn in enumerate(bd[c]):
            lo,hi=bd[c][gn]; sl[i,k]=min(g[i,k]-lo,hi-g[i,k])
    Ag=b.A_g.values.astype(bool); Disc=(q>=QMIN).all(axis=1)
    rel=np.where(Ag&Disc&(ts>=TAU))[0]
    S=shapley(g,b,bd,rel)
    top_sh=S.argmax(axis=1); top_cf=sl[rel].argmin(axis=1)
    ctx=b.context_label.values[rel]
    for c in sorted(set(ctx)):
        m=ctx==c
        vals,cnt=np.unique(top_sh[m],return_counts=True)
        conc_sh=cnt.max()/cnt.sum()
        vals2,cnt2=np.unique(top_cf[m],return_counts=True)
        conc_cf=cnt2.max()/cnt2.sum()
        rows.append(dict(domain=dom,context=c,n=int(m.sum()),
                         pool_ref_modal_share=round(float(conc_sh),3),
                         local_ref_modal_share=round(float(conc_cf),3)))
out=pd.DataFrame(rows); out.to_csv(D + "shapley_by_context.csv",index=False)
print(out.to_string(index=False))
print()
for dom in ["RLV","Healthcare"]:
    s=out[out.domain==dom]
    print(f"{dom}: havuz referansi baglam ici tepe payi {s.pool_ref_modal_share.min():.3f}-{s.pool_ref_modal_share.max():.3f}"
          f" | yerel referans {s.local_ref_modal_share.min():.3f}-{s.local_ref_modal_share.max():.3f}")
