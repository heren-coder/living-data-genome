import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
D="../data/"
a=pd.read_csv(D+"ablation_results.csv"); sg=pd.read_csv(D+"ablation_significance.csv")
INK="#1F3864"; ACC="#C55A11"; BK="#000000"
FULL={"RLV":dict(A=0.716169,SR=0.870083),"Healthcare":dict(A=0.749412,SR=0.849250)}
NAME={"RLV":{"S":"Speed","A":"Attention","D":"Density","E":"Environment"},
      "Healthcare":{"S":"Access escalation","A":"Audit coverage",
                    "D":"Record density","E":"Access control"}}
OFF={"RLV":{"D":(10,4,"left"),"S":(-8,-16,"right"),"A":(10,2,"left"),"E":(2,-18,"center")},
     "Healthcare":{"S":(10,4,"left"),"A":(-8,-16,"right"),"D":(10,-6,"left"),"E":(10,4,"left")}}
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,67.9*MM),dpi=600)
axes=[fig.add_axes([0.085,0.200,0.375,0.560]),fig.add_axes([0.585,0.200,0.375,0.560])]
for ax,dom in zip(axes,["RLV","Healthcare"]):
    d=a[a.domain==dom]
    for r in d.itertuples():
        dA=r.A_triad_mean-FULL[dom]["A"]; dS=r.SR-FULL[dom]["SR"]; dR=r.Rel_delta_vs_full
        col = ACC if dR<0 else INK
        sig = bool(sg[(sg.domain==dom)&(sg.gene_dropped==r.gene_dropped)&
                      (sg.metric=="A_triad")].holm_significant.iloc[0])
        ax.scatter([dA],[dS],s=64,marker="o",zorder=4,
                   facecolors=(col if sig else "none"),edgecolors=col,linewidths=1.4)
        lab=f"{NAME[dom][r.gene_dropped]}\n\u0394Rel {dR:+.3f}"
        ox,oy,ha=OFF[dom][r.gene_dropped]
        ax.annotate(lab,(dA,dS),textcoords="offset points",xytext=(ox,oy),
                    ha=ha,fontsize=6.4,color=col,linespacing=1.2)
    ax.axhline(0,color="#9A9A9A",linewidth=0.8); ax.axvline(0,color="#9A9A9A",linewidth=0.8)
    ax.set_xlabel("change in triadic alignment",fontsize=7.2,color=BK)
    ax.set_ylabel("change in admissibility survival",fontsize=7.2,color=BK)
    ax.tick_params(labelsize=6.8,colors=BK)
    ax.set_xlim(-0.165,0.075); ax.set_ylim(-0.004,0.062)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(BK); ax.spines['bottom'].set_color(BK)
    ax.set_title(dom,fontsize=7.8,color="#0F2545",fontweight="bold",pad=16)
fig.text(0.5,0.937,"dropping a gene inflates survival and costs alignment; only where the cost wins does the aggregate fall",
         ha="center",fontsize=7.4,color="#0F2545",fontweight="bold")
fig.text(0.5,0.045,"filled markers survive the Holm correction across the sixteen tests; orange marks the gene whose removal lowers the aggregate",
         ha="center",fontsize=6.6,color=BK,style="italic")
fig.savefig("Figure_11.png",facecolor="white")
print("ok")
