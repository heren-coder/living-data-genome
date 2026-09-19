"""LIME surrogate on the declared release rule, compared against the exact
interventional Shapley values and the counterfactual binding coordinate."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED

import itertools, sys
from math import factorial
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG
QMIN, TAU, STS = 0.50, 0.90, 8.0
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
NAMES = {"RLV": ["Speed","Attention","Density","Environment"],
         "Healthcare": ["AccessEscalationRate","AuditCoverage",
                        "ConcurrentRecordDensity","AccessControlRegime"]}
NBG = 200
RNG = np.random.default_rng(7)
LRNG = np.random.default_rng(11)
NLIME = 500

def load(dom):
    df = pd.read_csv(D+"scenarios.csv"); d = df[df.domain==dom]
    a = d[d.stage_t==1].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    b = d[d.stage_t==2].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    g1 = a[["g_S","g_A","g_D","g_E"]].values
    g  = b[["g_S","g_A","g_D","g_E"]].values
    q  = b[["q_S","q_A","q_D","q_E"]].values
    qn = q/q.sum(axis=1,keepdims=True)
    ts = np.exp(-((g-g1)*(STS*qn)*(g-g1)).sum(axis=1))
    return b,g,q,ts

def bounds(b,cfg):
    return {c: cfg.feasibility_bounds(c) for c in b.context_label.unique()}

def slacks(b,g,bd):
    sl=np.zeros_like(g)
    for i,c in enumerate(b.context_label):
        for k,gn in enumerate(bd[c]):
            lo,hi=bd[c][gn]; sl[i,k]=min(g[i,k]-lo, hi-g[i,k])
    return sl

def inside(X, ctx, bd):
    ok=np.ones(len(X),bool)
    for k,gn in enumerate(bd[ctx]):
        lo,hi=bd[ctx][gn]; ok &= (X[:,k]>=lo)&(X[:,k]<=hi)
    return ok

def shapley(g,b,bd,idx):
    subs=[s for r in range(5) for s in itertools.combinations(range(4),r)]
    pools={c: g[(b.context_label==c).values] for c in bd}
    S=np.zeros((len(idx),4))
    for n,i in enumerate(idx):
        ctx=b.context_label[i]
        smp=pools[ctx][RNG.integers(0,len(pools[ctx]),NBG)]
        v={}
        for s in subs:
            X=smp.copy()
            for k in s: X[:,k]=g[i,k]
            v[s]=inside(X,ctx,bd).mean()
        for k in range(4):
            tot=0.0
            for s in subs:
                if k in s: continue
                w=factorial(len(s))*factorial(3-len(s))/factorial(4)
                tot+=w*(v[tuple(sorted(s+(k,)))]-v[s])
            S[n,k]=tot
    return S

def lime(g,b,bd,idx):
    L=np.zeros((len(idx),4))
    for n,i in enumerate(idx):
        ctx=b.context_label[i]
        hw=np.array([(bd[ctx][gn][1]-bd[ctx][gn][0])/2 for gn in bd[ctx]])
        sig=0.5*hw
        Z=g[i]+LRNG.normal(0,sig,size=(NLIME,4))
        y=inside(Z,ctx,bd).astype(float)
        d=np.sqrt((((Z-g[i])/hw)**2).sum(axis=1))
        w=np.exp(-(d**2)/(0.75**2*4))
        Zs=(Z-g[i])/hw
        A=np.hstack([np.ones((NLIME,1)),Zs])
        W=np.diag(w)
        beta=np.linalg.solve(A.T@W@A+1e-6*np.eye(5), A.T@W@y)
        L[n]=beta[1:]
    return L

rows=[]
for dom in ["RLV","Healthcare"]:
    b,g,q,ts=load(dom); bd=bounds(b,CFG[dom]); sl=slacks(b,g,bd)
    Ag=b.A_g.values.astype(bool); Disc=(q>=QMIN).all(axis=1)
    Aimm=Ag&Disc&(ts>=TAU)
    idx=np.where(Aimm)[0]
    bind=sl[idx].argmin(axis=1)
    S=shapley(g,b,bd,idx); sh=np.abs(S).argmax(axis=1)
    L=lime(g,b,bd,idx);   lm=np.abs(L).argmax(axis=1)
    r=dict(domain=dom,n_released=len(idx),
           agree_cf_shapley=float((bind==sh).mean()),
           agree_cf_lime=float((bind==lm).mean()),
           agree_shapley_lime=float((sh==lm).mean()),
           all_three_agree=float(((bind==sh)&(sh==lm)).mean()),
           chance=0.25)
    for k in range(4):
        r[f"lime_top_share_{NAMES[dom][k]}"]=float((lm==k).mean())
    rows.append(r)
out=pd.DataFrame(rows)
out.to_csv("" + B + "lime_vs_shapley.csv",index=False)
print(out[["domain","n_released","agree_cf_shapley","agree_cf_lime",
           "agree_shapley_lime","all_three_agree"]].round(3).to_string(index=False))
