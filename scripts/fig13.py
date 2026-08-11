import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
k=pd.read_csv("../data/kappa_boundary.csv"); sc=pd.read_csv("../data/kappa_boundary_seedcheck.csv")
INK="#1F3864"; ACC="#C55A11"; BK="#000000"
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,69.6*MM),dpi=600)
a1=fig.add_axes([0.075,0.195,0.375,0.560])
a2=fig.add_axes([0.545,0.195,0.275,0.560])
a3=fig.add_axes([0.895,0.195,0.088,0.560])

BND={"RLV":16.3,"Healthcare":17.0}
for dom,col,mk in [("RLV",INK,"o"),("Healthcare",ACC,"s")]:
    s=k[k.domain==dom].sort_values("variance_pct")
    a1.plot(s.variance_pct,s.kappa_mean,color=col,marker=mk,markersize=3.6,linewidth=1.3,label=dom)
    a1.fill_between(s.variance_pct,s.kappa_mean-s.kappa_sd,s.kappa_mean+s.kappa_sd,
                    color=col,alpha=0.14,linewidth=0)
    a2.plot(s.variance_pct,s.cv,color=col,marker=mk,markersize=3.6,linewidth=1.3,label=dom)
    a1.axvline(BND[dom],color=col,linewidth=0.9,linestyle=(0,(3,2)),alpha=0.8)
    a2.axvline(BND[dom],color=col,linewidth=0.9,linestyle=(0,(3,2)),alpha=0.8)
a1.plot([5],[k[(k.domain=="RLV")&(k.variance_pct==5)].kappa_mean.iloc[0]],marker="o",
        markersize=7,markerfacecolor="none",markeredgecolor=BK,markeredgewidth=1.3,zorder=6)
a1.annotate("operating point",(5,k[(k.domain=="RLV")&(k.variance_pct==5)].kappa_mean.iloc[0]),
            textcoords="offset points",xytext=(9,10),fontsize=6.6,color=BK)
a1.set_xlabel("inter-node calibration variance (%)",fontsize=7.2,color=BK)
a1.set_ylabel("agreement coefficient",fontsize=7.2,color=BK)
a1.set_ylim(-0.05,1.02); a1.tick_params(labelsize=6.8,colors=BK)
for sp in ("top","right"): a1.spines[sp].set_visible(False)
a1.spines['left'].set_color(BK); a1.spines['bottom'].set_color(BK)
a1.legend(fontsize=6.8,frameon=False,loc="upper right",labelcolor=BK)
a1.set_title("agreement declines, then scatters",
             fontsize=7.6,color="#0F2545",fontweight="bold",pad=8)

a2.axhline(0.5,color=BK,linewidth=1.0,linestyle=(0,(4,3)))
a2.text(3.4,0.42,"declared criterion:\nspread reaches half the mean",
        fontsize=6.2,color=BK,va="top",linespacing=1.25)
a2.set_xlabel("inter-node calibration variance (%)",fontsize=7.2,color=BK)
a2.set_ylabel("spread relative to the mean",fontsize=7.2,color=BK)
a2.set_xlim(1,32); a2.set_ylim(0,1.6); a2.tick_params(labelsize=6.8,colors=BK)
for sp in ("top","right"): a2.spines[sp].set_visible(False)
a2.spines['left'].set_color(BK); a2.spines['bottom'].set_color(BK)
a2.set_title("where the boundary is read",fontsize=7.6,color="#0F2545",fontweight="bold",pad=8)

for dom,col,mk in [("RLV",INK,"o"),("Healthcare",ACC,"s")]:
    s=sc[sc.domain==dom]
    a3.plot(s.n_seeds,s.boundary_pct,color=col,marker=mk,markersize=3.6,linewidth=1.3)
a3.set_xlabel("seeds",fontsize=7.0,color=BK)
a3.set_ylabel("boundary (%)",fontsize=6.8,color=BK,labelpad=1)
a3.set_ylim(14.5,18.5); a3.set_xticks([5,15,25])
a3.tick_params(labelsize=6.6,colors=BK)
for sp in ("top","right"): a3.spines[sp].set_visible(False)
a3.spines['left'].set_color(BK); a3.spines['bottom'].set_color(BK)
a3.set_title("and how\nstable it is",fontsize=7.0,color="#0F2545",fontweight="bold",pad=8,
             linespacing=1.2)
fig.text(0.5,0.045,"twenty-five seeds per point; the boundary moves by less than half a point between five and twenty-five seeds",
         ha="center",fontsize=6.6,color=BK,style="italic")
fig.savefig("Figure_13.png",facecolor="white")
print("ok")
