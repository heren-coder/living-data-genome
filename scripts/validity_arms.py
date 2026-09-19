"""Convergent and discriminant response of the aggregate.

Convergent arms damage exactly one component's target failure mode and the
aggregate must fall. Discriminant arms apply transformations that change no
institutional fact and the aggregate must not move beyond a declared band.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
import rel_computation as R

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
import rel_computation as R
CFG={"RLV":RLV_CONFIG,"Healthcare":HEALTHCARE_CONFIG}
G=["g_S","g_A","g_D","g_E"]; Q=["q_S","q_A","q_D","q_E"]; K=["S","A","D","E"]
SEEDS=[0,1,2,3,4]
BAND=0.01     # declared invariance band for the discriminant arms

def admissible(row,cfg):
    b=cfg.feasibility_bounds(row.context_label)
    g=[row.g_S,row.g_A,row.g_D,row.g_E]
    return all(b[k][0]<=g[i]<=b[k][1] for i,k in enumerate(K))

def score(df,cfg,dom,seed,gc=1.0):
    s,_,_=R.compute_rel(dom,df,cfg,gc_placeholder=gc,seed=RNG_SEED+seed)
    return dict(A=s["A_triad"],TS=s["TS_mean"],SR=s["SR_mean"],GC=gc,Rel=s["Rel_mean"])

# ---------------- convergent arms ----------------
def c1_misalign(df,cfg,rng,strength=0.35):
    """Semantic misalignment: scramble the repair-stage coordinate within the
    admissible band, leaving admissibility and the transitions' magnitude intact."""
    d=df.copy()
    m=d.stage_t==2
    idx=d.index[m]
    pick=rng.choice(idx,size=int(strength*len(idx)),replace=False)
    perm=rng.permutation(pick)
    d.loc[pick,G]=d.loc[perm,G].values
    return d

def c2_unstable(df,cfg,rng,extra=0.10):
    """Instability: enlarge the mutation displacement without moving genesis or repair."""
    d=df.copy(); m=d.stage_t==1
    d.loc[m,G]=np.clip(d.loc[m,G].values+rng.normal(0,extra,size=(m.sum(),4)),0,1)
    return d

def c3_unreleasable(df,cfg,rng,shrink=0.10):
    """Loss of releasability: push repair-stage candidates outward so fewer survive."""
    d=df.copy(); m=d.stage_t==2
    ctr=np.array([[np.mean(cfg.feasibility_bounds(c)[k]) for k in K]
                  for c in d.loc[m,"context_label"]])
    g=d.loc[m,G].values
    d.loc[m,G]=np.clip(g+shrink*np.sign(g-ctr),0,1)
    d.loc[m,"A_g"]=[admissible(r,cfg) for r in d[m].itertuples()]
    return d

# ---------------- discriminant arms ----------------
def d1_permute(df,cfg,rng):
    """Relabel candidates within each family, applying one permutation to all three
    revisions so that every trajectory keeps its own identity; the family is a set of
    trajectories and its ordering carries no institutional content."""
    d=df.copy()
    new=np.empty(len(d),dtype=int)
    for ev,g in d.groupby("event_id"):
        ids=np.sort(g.candidate_id.unique())
        perm=rng.permutation(len(ids))
        mp={old:int(perm[i]) for i,old in enumerate(ids)}
        new[g.index.values]=[mp[c] for c in g.candidate_id.values]
    d["candidate_id"]=new
    return d.sort_values(["event_id","candidate_id","stage_t"]).reset_index(drop=True)

def d2_quality_scale(df,cfg,rng,c=0.60):
    """Rescale all quality metadata by a constant; the weighting uses the normalized
    vector, so no institutional statement changes."""
    d=df.copy(); d[Q]=d[Q].values*c
    return d

def d3_relabel(df,cfg,rng):
    """Rename the operating contexts under a bijection, applied consistently."""
    d=df.copy()
    cats=list(cfg.context_categories)
    mapping={c:cats[(i+1)%len(cats)] for i,c in enumerate(cats)}
    inv={v:k for k,v in mapping.items()}
    d["context_label"]=[mapping[c] for c in d.context_label]
    return d,inv

class Renamed:
    """A config whose context names are permuted to match arm D3."""
    def __init__(self,cfg,inv):
        self._c=cfg; self._inv=inv
        self.context_categories=[k for k in cfg.context_categories]
        self.raw_trace_dim=cfg.raw_trace_dim; self.gene_labels=cfg.gene_labels
        self.feasibility_tolerance=cfg.feasibility_tolerance; self.name=cfg.name
        self.context_means={c:cfg.context_means[inv[c]] for c in cfg.context_categories}
        self.pi={c:cfg.pi[inv[c]] for c in cfg.context_categories}
    def feasibility_bounds(self,c): return self._c.feasibility_bounds(self._inv[c])


# ---------------- inflation arm ----------------
def x1_enforce(df,cfg,rng):
    """Enforce admissibility by construction: project every repair-stage candidate
    onto the admissible region instead of measuring whether it landed there.
    Section 2.5.4 identifies this as the reading to avoid."""
    d=df.copy(); m=d.stage_t==2
    g=d.loc[m,G].values.copy()
    for i,c in enumerate(d.loc[m,"context_label"]):
        b=cfg.feasibility_bounds(c)
        for j,k in enumerate(K):
            lo,hi=b[k]; g[i,j]=min(max(g[i,j],lo),hi)
    d.loc[m,G]=g
    d.loc[m,"A_g"]=True
    return d

rows=[]
sc=pd.read_csv(D+"scenarios.csv")
for dom,cfg in CFG.items():
    base=sc[sc.domain==dom].reset_index(drop=True)
    for seed in SEEDS:
        rng=np.random.default_rng(RNG_SEED+101*seed)
        b=score(base,cfg,dom,seed)
        rows.append(dict(domain=dom,seed=seed,arm="baseline",kind="-",**b))
        for name,fn in [("misalignment",c1_misalign),("instability",c2_unstable),
                        ("releasability",c3_unreleasable)]:
            d=fn(base,cfg,np.random.default_rng(RNG_SEED+101*seed+7))
            rows.append(dict(domain=dom,seed=seed,arm=name,kind="convergent",
                             **score(d,cfg,dom,seed)))
        # lineage: a share of revocations left without a disposition event
        for frac in (0.30,):
            rows.append(dict(domain=dom,seed=seed,arm="lineage",kind="convergent",
                             **score(base,cfg,dom,seed,gc=1.0-frac)))
        for name,fn in [("candidate order",d1_permute),("quality scale",d2_quality_scale)]:
            d=fn(base,cfg,np.random.default_rng(RNG_SEED+101*seed+11))
            rows.append(dict(domain=dom,seed=seed,arm=name,kind="discriminant",
                             **score(d,cfg,dom,seed)))
        d=x1_enforce(base,cfg,np.random.default_rng(RNG_SEED+101*seed+17))
        rows.append(dict(domain=dom,seed=seed,arm="enforced admissibility",kind="inflation",
                         **score(d,cfg,dom,seed)))
        d,inv=d3_relabel(base,cfg,np.random.default_rng(RNG_SEED+101*seed+13))
        rows.append(dict(domain=dom,seed=seed,arm="context naming",kind="discriminant",
                         **score(d,Renamed(cfg,inv),dom,seed)))
out=pd.DataFrame(rows); out.to_csv(D + "validity_arms_raw.csv",index=False)
agg=out.groupby(["domain","arm","kind"]).agg(
    A=("A","mean"),TS=("TS","mean"),SR=("SR","mean"),GC=("GC","mean"),
    Rel=("Rel","mean"),Rel_sd=("Rel","std")).reset_index()
base=agg[agg.arm=="baseline"].set_index("domain")
agg["dRel"]=[r.Rel-base.loc[r.domain,"Rel"] for r in agg.itertuples()]
agg.to_csv(D + "validity_arms.csv",index=False)
print(agg.round(4).to_string(index=False))
