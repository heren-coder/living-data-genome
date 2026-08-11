import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
e=pd.read_csv("../data/sweep_eps.csv")
s=pd.read_csv("../data/sweep_sTS.csv") if __import__("os").path.exists("../data/sweep_sTS.csv") else None
INK="#1F3864"; ACC="#C55A11"; BK="#000000"
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,63*MM),dpi=600)
a1=fig.add_axes([0.080,0.280,0.370,0.545])
a2=fig.add_axes([0.590,0.280,0.370,0.545])
d=e[e.domain=="RLV"].sort_values("eps_P")
a1.fill_between(d.eps_P,d.TS_p05,d.TS_p95,color=INK,alpha=0.15,linewidth=0)
a1.plot(d.eps_P,d.TS_mean,color=INK,linewidth=1.6)
env=(0.42,0.65)
a1.axvspan(env[0],env[1],facecolor=ACC,alpha=0.12,zorder=0)
a1.text((env[0]+env[1])/2,0.93,"displacements the\ngenerator produced",ha="center",va="top",
        fontsize=6.4,color=ACC,linespacing=1.2)
op=d[np.isclose(d.eps_P,0.25)]
a1.plot(op.eps_P,op.TS_mean,marker="o",markersize=7,markerfacecolor="none",
        markeredgecolor=BK,markeredgewidth=1.4,zorder=5)
a1.annotate("declared radius 0.25",(0.25,float(op.TS_mean.iloc[0])),
            textcoords="offset points",xytext=(0,-34),ha="center",fontsize=6.6,color=BK)
a1.set_xlabel("declared perturbation radius",fontsize=7.4,color=BK)
a1.set_ylabel("transition stability",fontsize=7.4,color=BK)
a1.set_ylim(0.30,1.02)
a1.set_title("what the protocol is entitled to probe",fontsize=7.6,color="#0F2545",
             fontweight="bold",pad=8)
if s is not None:
    for dom,col,mk in [("RLV",INK,"o"),("Healthcare",ACC,"s")]:
        t=s[s.domain==dom].sort_values("scale")
        lw = 3.0 if dom=="RLV" else 1.3
        st = "-" if dom=="RLV" else (0,(3,2))
        a2.plot(t.scale,t.TS_mean,color=col,marker=mk,markersize=3.4,linewidth=lw,
                linestyle=st,alpha=(0.45 if dom=="RLV" else 1.0),label=dom)
        if "domain" not in s.columns: break
    a2.axvline(8.0,color=BK,linewidth=0.9,linestyle=(0,(3,2)))
    a2.text(8.4,a2.get_ylim()[0]+0.005,"declared 8",fontsize=6.6,color=BK)
    a2.set_xscale("log",base=2); a2.set_xticks([1,2,4,8,16,32])
    a2.set_xticklabels(["1","2","4","8","16","32"])
    a2.legend(fontsize=6.8,frameon=False,loc="upper right",labelcolor=BK)
a2.set_xlabel("stability scale",fontsize=7.4,color=BK)
a2.set_ylabel("transition stability",fontsize=7.4,color=BK)
a2.set_title("and the scale it reads the result on",fontsize=7.6,color="#0F2545",
             fontweight="bold",pad=8)
for a in (a1,a2):
    a.tick_params(labelsize=6.8,colors=BK)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    a.spines['left'].set_color(BK); a.spines['bottom'].set_color(BK)
fig.text(0.5,0.085,"left: mean stability over perturbations drawn from the declared ball, with the fifth to ninety-fifth percentile shaded",
         ha="center",fontsize=6.5,color=BK,style="italic")
fig.text(0.5,0.035,"right: the two configurations coincide to within 0.0003 across the range and are drawn one over the other",
         ha="center",fontsize=6.5,color=BK,style="italic")
fig.savefig("Figure_18.png",facecolor="white")
print("ok")
