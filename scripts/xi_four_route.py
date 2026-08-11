"""Four-route outcome across the mismatch tolerance, Equation (2.20).

Writes ../data/sweep_xi_four_route.csv, the input to Figure 14 panels one and two.
Extracted from the routing script of an earlier revision; the computation is
unchanged, only the plotting was removed.
"""
import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon
from generator import _simplex
D='../data/'
df=pd.read_csv(D+'scenarios.csv'); pil=pd.read_csv(D+'pi_lookup.csv')
def pim(dom): return {r["context_label"]:np.array([r["pi_S"],r["pi_A"],r["pi_D"],r["pi_E"]]) for _,r in pil[pil.domain==dom].iterrows()}
def jsd(p,q): return float(jensenshannon(p,q)**2)
ETA=0.010; XI_REF=0.030
XI=np.array([0.002,0.005,0.0075,0.010,0.015,0.020,0.025,0.030,0.040,0.050,0.065,0.080])
C={"proceed":"#0E6A60","context_review":"#54307A","quarantine_repair":"#A8481A"}
L={"proceed":"proceed","context_review":"context review","quarantine_repair":"quarantine / repair"}
route={}; oc={}
for dom in ["RLV","Healthcare"]:
    P=pim(dom); labels=sorted(P)
    d=df[(df.domain==dom)&(df.stage_t==2)].reset_index(drop=True)
    rho=np.vstack([_simplex(r) for r in d[["g_S","g_A","g_D","g_E"]].values])
    Ag=d.A_g.values.astype(bool)
    Xi0=np.array([jsd(rho[i],P[d.context_label[i]]) for i in range(len(d))])
    bx=np.array([min(jsd(rho[i],P[c]) for c in labels) for i in range(len(d))])
    Actx=(Xi0-bx)>=ETA
    route[dom]=pd.DataFrame([{ "xi":x,"proceed":(Ag&(Xi0<=x)).mean(),
        "context_review":(Ag&(Xi0>x)&Actx).mean(),"quarantine_repair":(Ag&(Xi0>x)&~Actx).mean()} for x in XI])
    pos,neg=[],[]
    for s in range(20):
        rng=np.random.default_rng(s); mis=rng.random(len(d))<0.20
        decl=[rng.choice([c for c in labels if c!=t]) if m else t for t,m in zip(d.context_label,mis)]
        X=np.array([jsd(rho[i],P[decl[i]]) for i in range(len(d))])
        pos.append(X[Ag&mis]); neg.append(X[Ag&~mis])
    pos=np.concatenate(pos); neg=np.concatenate(neg)
    g=np.linspace(0.001,0.12,400)
    oc[dom]=(g,np.array([(pos>t).mean() for t in g]),np.array([(neg>t).mean() for t in g]))
pd.concat([route[d].assign(domain=d) for d in route]).to_csv("../data/sweep_xi_four_route.csv",index=False)
print("Saved: ../data/sweep_xi_four_route.csv")
