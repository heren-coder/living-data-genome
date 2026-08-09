import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv("validity_arms.csv")
FALL="#C55A11"; HOLD="#8A8A8A"; RISE="#2E7D32"; BK="#000000"; INK="#0F2545"
ARMS=[("releasability","loss of releasability",FALL),
      ("misalignment","semantic misalignment",FALL),
      ("lineage","lineage breakdown",FALL),
      ("instability","instability",FALL),
      ("__flat__","three transformations   0.000",HOLD),
      ("enforced admissibility","admissibility enforced,\nnot measured",RISE)]
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,82*MM),dpi=600)
axs=[fig.add_axes([0.085,0.200,0.300,0.620]),fig.add_axes([0.560,0.200,0.300,0.620])]
for ax,dom in zip(axs,["RLV","Healthcare"]):
    s=d[d.domain==dom].set_index("arm")
    base=float(s.loc["baseline","Rel"])
    ax.plot([0,1],[base,base],color=HOLD,linewidth=0.7,linestyle=(0,(2,3)),zorder=1)
    for arm,lab,col in ARMS:
        v = base if arm=="__flat__" else float(s.loc[arm,"Rel"])
        ax.plot([0,1],[base,v],color=col,linewidth=1.7,zorder=3,
                solid_capstyle="round",alpha=0.95)
        ax.plot([1],[v],marker="o",markersize=4.6,color=col,zorder=4)
        dv = 0.0 if arm=="__flat__" else v-base
        txt = lab if arm=="__flat__" else f"{lab}   {dv:+.3f}"
        dy = 6 if arm=="__flat__" else (-9 if arm=="instability" else (3 if arm=="enforced admissibility" else 0))
        ax.annotate(txt,(1,v),textcoords="offset points",xytext=(9,dy),
                    va="center",ha="left",fontsize=6.6,color=col,linespacing=1.15)
    ax.plot([0],[base],marker="o",markersize=5.2,color=BK,zorder=5)
    ax.annotate(f"undamaged\n{base:.3f}",(0,base),textcoords="offset points",xytext=(0,-14),
                va="top",ha="center",fontsize=6.8,color=BK,fontweight="bold",linespacing=1.2)
    ax.set_xlim(-0.30,2.55); ax.set_ylim(0.63,0.94)
    ax.set_xticks([]); ax.tick_params(axis='y',labelsize=6.8,colors=BK)
    ax.set_ylabel("aggregate" if dom=="RLV" else "",fontsize=7.4,color=BK)
    for sp in ("top","right","bottom"): ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(BK)
    ax.set_title(dom,fontsize=7.8,color=INK,fontweight="bold",pad=8)
fig.text(0.5,0.985,"the aggregate falls under every failure mode it targets, holds exactly under\ntransformations that change nothing, and rises when the measurement is removed",
         ha="center",va="top",fontsize=7.4,color=INK,fontweight="bold",linespacing=1.4)
fig.text(0.5,0.080,"removing the measurement gains 0.048 on average, four times what hiding a real failure mode gains",
         ha="center",fontsize=7.0,color=RISE,fontweight="bold")
fig.text(0.5,0.030,"means over five seeds against the undamaged pipeline on the same events; the three transformations coincide exactly and are drawn as one line",
         ha="center",fontsize=6.5,color=BK,style="italic")
fig.savefig("Figure_16.png",facecolor="white")
print("ok")
