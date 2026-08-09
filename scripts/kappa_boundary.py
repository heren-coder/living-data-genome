"""Where does the agreement estimate stop being stable, and is that boundary
itself stable in the number of seeds?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
D="../data/"
df=pd.read_csv(D+"scenarios.csv")
CFG={"RLV":RLV_CONFIG,"Healthcare":HEALTHCARE_CONFIG}
K=["S","A","D","E"]
VARP=[3,5,9,12,15,17,20,22,25,30,40]
NSEED=25

def kappa(calls):
    n,m=calls.shape; p=calls.mean()
    Pi=((calls.sum(1)**2+(m-calls.sum(1))**2)-m)/(m*(m-1))
    Pe=p**2+(1-p)**2
    return (Pi.mean()-Pe)/(1-Pe) if Pe<1 else 1.0

rows=[]
for dom,cfg in CFG.items():
    pool=df[(df.domain==dom)&(df.stage_t==2)].head(150)
    tol=cfg.feasibility_tolerance
    bnds={c:cfg.feasibility_bounds(c) for c in pool.context_label.unique()}
    G=pool[["g_S","g_A","g_D","g_E"]].values; ctx=pool.context_label.values
    for vp in VARP:
        ks=[]
        for s in range(NSEED):
            rng=np.random.default_rng(RNG_SEED+7919*s+vp)
            calls=np.zeros((len(pool),4),bool)
            for j in range(4):
                dt=rng.normal(0,tol*vp/100.0)
                for i in range(len(pool)):
                    b=bnds[ctx[i]]
                    calls[i,j]=all(b[k][0]-dt<=G[i,q]<=b[k][1]+dt for q,k in enumerate(K))
            ks.append(kappa(calls))
        ks=np.array(ks)
        rows.append(dict(domain=dom,variance_pct=vp,n_seeds=NSEED,
                         kappa_mean=float(ks.mean()),kappa_sd=float(ks.std(ddof=1)),
                         cv=float(ks.std(ddof=1)/ks.mean()),
                         samples=";".join(f"{x:.4f}" for x in ks)))
out=pd.DataFrame(rows)
out.drop(columns=["samples"]).to_csv("kappa_boundary.csv",index=False)
print(out[["domain","variance_pct","kappa_mean","kappa_sd","cv"]].round(3).to_string(index=False))

def boundary(sub):
    sub=sub.sort_values("variance_pct")
    v=sub.variance_pct.values; c=sub.cv.values
    i=np.where(c>0.5)[0]
    if not len(i): return None
    j=i[0]
    if j==0: return float(v[0])
    return float(np.interp(0.5,[c[j-1],c[j]],[v[j-1],v[j]]))

print("\n=== sinir, 25 tohum ===")
for dom in CFG:
    print(f"  {dom}: {boundary(out[out.domain==dom]):.1f}%")

print("\n=== ikinci kontrol: sinirin tohum sayisina duyarliligi ===")
chk=[]
for dom in CFG:
    s=out[out.domain==dom]
    for n in (5,10,15,25):
        sub=[]
        for _,r in s.iterrows():
            x=np.array([float(t) for t in r.samples.split(";")])[:n]
            sub.append(dict(variance_pct=r.variance_pct,cv=x.std(ddof=1)/x.mean()))
        b=boundary(pd.DataFrame(sub))
        chk.append(dict(domain=dom,n_seeds=n,boundary_pct=round(b,1) if b else None))
c=pd.DataFrame(chk); c.to_csv("kappa_boundary_seedcheck.csv",index=False)
print(c.to_string(index=False))
