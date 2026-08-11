"""Joint sweep of the two protocol constants that bound where the mechanism works:
governance tightening magnitude and inter-node calibration variance."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
FIGDIR = _os.path.join(_HERE, "..", "figures") + _os.sep
import itertools, numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import REL_WEIGHTS
from day5_governance import tightened_bounds, is_admissible

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import REL_WEIGHTS
from day5_governance import tightened_bounds, is_admissible
df=pd.read_csv(D+"scenarios.csv"); rel=pd.read_csv(D+"rel_summary.csv").set_index("domain")
CFG={"RLV":RLV_CONFIG,"Healthcare":HEALTHCARE_CONFIG}
TIGHT=[0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20]
VARP=[3,5,9,15,20,25,30]
SEEDS=[0,1,2,3,4]
K=["S","A","D","E"]

def gen1(sub):
    r=sub[sub.stage_t==2]
    return r[r.A_g].sort_values("candidate_id").groupby("event_id").first().reset_index()

def arms(g1,cfg,tight,rng):
    kept=[];rev=[]
    for _,row in g1.iterrows():
        c=row["context_label"]; g=np.array([row.g_S,row.g_A,row.g_D,row.g_E])
        (kept if is_admissible(g,tightened_bounds(cfg,c,tight)) else rev).append(row)
    n=len(g1); ok=0
    for row in rev:
        c=row["context_label"]; cp=tightened_bounds(cfg,c,tight)
        g0=np.array([row.g_S,row.g_A,row.g_D,row.g_E])
        m=np.clip(g0+rng.normal(0,0.14,4),0,1)
        ctr=np.array([(cp[k][0]+cp[k][1])/2 for k in K])
        rp=np.clip(m+0.55*(ctr-m)+rng.normal(0,0.042,4),0,1)
        ok+=int(is_admissible(rp,cp))
    sr_d=len(kept)/n; gc_d=1-len(rev)/n
    sr_r=(len(kept)+ok)/n
    return sr_d,gc_d,sr_r

def relscore(a,ts,sr,gc):
    return (a**REL_WEIGHTS["A_triad"])*(ts**REL_WEIGHTS["TS"])*(max(sr,1e-6)**REL_WEIGHTS["SR"])*(max(gc,1e-6)**REL_WEIGHTS["GC"])

def kappa(calls):
    n,m=calls.shape; p=calls.mean()
    Pi=((calls.sum(1)**2+(m-calls.sum(1))**2)-m)/(m*(m-1))
    Pbar=Pi.mean(); Pe=p**2+(1-p)**2
    return (Pbar-Pe)/(1-Pe) if Pe<1 else 1.0

rows=[]
for dom,cfg in CFG.items():
    sub=df[df.domain==dom]; g1=gen1(sub)
    a=float(rel.loc[dom,"A_triad"]); ts=float(rel.loc[dom,"TS_mean"])
    pool=sub[(sub.stage_t==2)].head(150)
    tol=cfg.feasibility_tolerance
    for tight,vp in itertools.product(TIGHT,VARP):
        gaps=[];kaps=[]
        for s in SEEDS:
            rng=np.random.default_rng(RNG_SEED+1000*s+int(tight*1000)+vp)
            sr_d,gc_d,sr_r=arms(g1,cfg,tight,rng)
            gaps.append(relscore(a,ts,sr_r,1.0)-relscore(a,ts,sr_d,gc_d))
            calls=np.zeros((len(pool),4),bool)
            for j in range(4):
                dt=rng.normal(0,tol*vp/100.0)
                for i,row in enumerate(pool.itertuples()):
                    b=cfg.feasibility_bounds(row.context_label)
                    g=[row.g_S,row.g_A,row.g_D,row.g_E]
                    calls[i,j]=all(b[k][0]-dt<=g[q]<=b[k][1]+dt for q,k in enumerate(K))
            kaps.append(kappa(calls))
        rows.append(dict(domain=dom,tightening=tight,variance_pct=vp,
                         rel_gap_mean=float(np.mean(gaps)),rel_gap_sd=float(np.std(gaps,ddof=1)),
                         kappa_mean=float(np.mean(kaps)),kappa_sd=float(np.std(kaps,ddof=1))))
out=pd.DataFrame(rows); out.to_csv(D + "operating_map.csv",index=False)
print("nokta:",len(out))
for dom in CFG:
    s=out[out.domain==dom]
    print(f"\n== {dom} : Rel farki (satir=sikilastirma, sutun=degiskenlik %)")
    print(s.pivot(index="tightening",columns="variance_pct",values="rel_gap_mean").round(3).to_string())
    print(f"-- kappa degisim katsayisi")
    p=s.copy(); p["cv"]=p.kappa_sd/p.kappa_mean
    print(p.pivot(index="tightening",columns="variance_pct",values="cv").round(2).to_string())
