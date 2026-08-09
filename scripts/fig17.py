import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv("../data/sweep_delta.csv")
INK="#1F3864"; ACC="#C55A11"; GRN="#2E7D32"; BK="#000000"
OP={"RLV":0.33,"Healthcare":0.31}
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,66*MM),dpi=600)
a1=fig.add_axes([0.075,0.235,0.375,0.585])
a2=fig.add_axes([0.585,0.235,0.375,0.585])
STY={"adm_genesis":("genesis",":"),"adm_mutation":("mutation","--"),"adm_repair":("repair","-")}
for dom,col,po,do in [("RLV",INK,(-8,-13),(11,4)),("Healthcare",ACC,(-8,10),(6,-14))]:
    s=d[d.domain==dom].sort_values("delta")
    for c,(lab,ls) in STY.items():
        a1.plot(s.delta,s[c],color=col,linestyle=ls,linewidth=1.3)
    a1.axvline(OP[dom],color=col,linewidth=0.9,linestyle=(0,(3,2)),alpha=0.75)
    a2.plot(s.delta,s.recovery,color=col,linewidth=1.5)
    k=s.recovery.idxmax()
    a2.plot([s.loc[k,"delta"]],[s.loc[k,"recovery"]],marker="o",markersize=7,
            markerfacecolor="none",markeredgecolor=col,markeredgewidth=1.4,zorder=5)
    op=s[np.isclose(s.delta,OP[dom])]
    a2.plot(op.delta,op.recovery,marker="o",markersize=4.6,color=col,zorder=6)
    a2.annotate(f"peak {s.loc[k,'delta']:.2f}",(s.loc[k,"delta"],s.loc[k,"recovery"]),
                textcoords="offset points",xytext=po,ha="right",fontsize=6.4,color=col)
    a2.annotate(f"declared {OP[dom]:.2f}",(float(op.delta.iloc[0]),float(op.recovery.iloc[0])),
                textcoords="offset points",xytext=do,ha="left",fontsize=6.4,color=col)
for lab,ls in [("genesis",":"),("mutation","--"),("repair","-")]:
    a1.plot([],[],color=BK,linestyle=ls,linewidth=1.3,label=lab)
a1.legend(fontsize=6.6,frameon=False,loc="lower right",labelcolor=BK,ncol=3,
          handlelength=1.8,columnspacing=1.0)
a1.plot([],[],color=INK,linewidth=1.3,label="RLV"); a1.plot([],[],color=ACC,linewidth=1.3)
a1.set_xlabel("feasibility tolerance",fontsize=7.4,color=BK)
a1.set_ylabel("admissibility rate",fontsize=7.4,color=BK)
a1.set_title("the dip and the recovery persist at every tolerance",fontsize=7.6,
             color="#0F2545",fontweight="bold",pad=8)
a2.set_xlabel("feasibility tolerance",fontsize=7.4,color=BK)
a2.set_ylabel("recovery at repair",fontsize=7.4,color=BK); a2.set_ylim(0.02,0.42)
a2.set_title("and the declared point is past the peak",fontsize=7.6,
             color="#0F2545",fontweight="bold",pad=8)
for a in (a1,a2):
    a.tick_params(labelsize=6.8,colors=BK)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    a.spines['left'].set_color(BK); a.spines['bottom'].set_color(BK)
fig.text(0.5,0.042,"solid, dashed and dotted lines are the three revisions; blue is the red-light violation configuration and orange the healthcare vignette",
         ha="center",fontsize=6.6,color=BK,style="italic")
fig.savefig("Figure_17.png",facecolor="white")
print("ok")
