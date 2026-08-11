"""Section 3.8: sensitivity of the artifact gate and immune condition to the
two constants they declare, q_min (Eq. 2.24) and tau (Eq. 2.16).
Outputs: data/sweep_gate.csv
"""
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG
D="../data/"; STS=8.0
CFG={"RLV":RLV_CONFIG,"Healthcare":HEALTHCARE_CONFIG}
df=pd.read_csv(D+"scenarios.csv"); rows=[]
for dom in ["RLV","Healthcare"]:
    d=df[df.domain==dom]
    a=d[d.stage_t==1].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    b=d[d.stage_t==2].sort_values(["event_id","candidate_id"]).reset_index(drop=True)
    g1=a[["g_S","g_A","g_D","g_E"]].values; g=b[["g_S","g_A","g_D","g_E"]].values
    q=b[["q_S","q_A","q_D","q_E"]].values; qn=q/q.sum(axis=1,keepdims=True)
    ts=np.exp(-((g-g1)*(STS*qn)*(g-g1)).sum(axis=1))
    Ag=b.A_g.values.astype(bool); ev=b.event_id.values
    for qm in [0.0,0.20,0.35,0.50,0.65,0.80]:
        for tau in [0.0,0.70,0.80,0.90,0.95,0.98]:
            AR=Ag&(q>=qm).all(axis=1); Ai=AR&(ts>=tau)
            e=pd.DataFrame({"e":ev,"a":Ai}).groupby("e").a.max().mean()
            rows.append(dict(domain=dom,q_min=qm,tau=tau,A_R=AR.mean(),
                             A_imm=Ai.mean(),event_level=e))
out=pd.DataFrame(rows); out.to_csv(D+"sweep_gate.csv",index=False)
for dom in ["RLV","Healthcare"]:
    s=out[(out.domain==dom)&(out.tau==0.90)]
    print(f"{dom}  tau=0.90, q_min 0->0.80 : A_imm "+" ".join(f"{v:.3f}" for v in s.A_imm))
    s=out[(out.domain==dom)&(out.q_min==0.50)]
    print(f"{dom}  q_min=0.50, tau 0->0.98: A_imm "+" ".join(f"{v:.3f}" for v in s.A_imm))
print("Saved: data/sweep_gate.csv")
