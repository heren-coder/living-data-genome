import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
D="../data/"
r=pd.read_csv(D+"sweep_xi_four_route.csv"); oc=pd.read_csv(D+"xi_operating_characteristic.csv")
et=pd.read_csv(D+"a1_eta_sweep.csv")
INK="#1F3864"; ACC="#C55A11"; PUR="#7B3FA0"; GRN="#2E7D32"; BK="#000000"
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,66*MM),dpi=600)
W=0.185; Y=0.250; H=0.560
ax1=fig.add_axes([0.062,Y,W,H]); ax2=fig.add_axes([0.315,Y,W,H])
ax3=fig.add_axes([0.560,Y,W,H]); ax4=fig.add_axes([0.800,Y,0.175,H])

for dom,ls in [("RLV","-"),("Healthcare","--")]:
    s=r[r.domain==dom].sort_values("xi")
    ax1.plot(s.xi,s.proceed,color=GRN,linestyle=ls,linewidth=1.4,marker=None)
    ax1.plot(s.xi,s.quarantine_repair,color=ACC,linestyle=ls,linewidth=1.4)
    ax2.plot(s.xi,s.context_review*100,color=PUR,linestyle=ls,linewidth=1.4,marker="o",markersize=2.6)
ax1.axvline(0.030,color=BK,linewidth=0.9,linestyle=(0,(3,2)))
ax1.text(0.032,0.50,"declared",fontsize=6.2,color=BK)
ax1.text(0.052,0.72,"proceed",fontsize=6.6,color=GRN)
ax1.text(0.042,0.16,"quarantine\nor repair",fontsize=6.6,color=ACC,linespacing=1.2)
ax1.set_xlabel("mismatch tolerance",fontsize=7.0,color=BK)
ax1.set_ylabel("share of candidates",fontsize=7.0,color=BK)
ax1.set_title("proceed and repair",fontsize=7.2,color="#0F2545",fontweight="bold",pad=6)

ax2.axvline(0.030,color=BK,linewidth=0.9,linestyle=(0,(3,2)))
ax2.set_xlabel("mismatch tolerance",fontsize=7.0,color=BK)
ax2.set_ylabel("context review (%)",fontsize=7.0,color=BK)
ax2.set_title("context review, own scale",fontsize=7.2,color="#0F2545",fontweight="bold",pad=6)
ax2.text(0.081,4.30,"tightest 4.2%\nloosest 0.1%",fontsize=6.4,color=PUR,
         ha="right",va="top",linespacing=1.25)

for dom,col,ls in [("RLV",INK,"-"),("Healthcare",ACC,"--")]:
    s=oc[oc.domain==dom].sort_values("false_flag")
    ax3.plot(s.false_flag,s.detection,color=col,linestyle=ls,linewidth=1.4,label=dom)
    p=s[np.isclose(s.xi,0.030)]
    if len(p): ax3.plot(p.false_flag,p.detection,marker="o",markersize=6,markerfacecolor="none",
                        markeredgecolor=col,markeredgewidth=1.4)
ax3.plot([0,1],[0,1],color="#9A9A9A",linewidth=0.8,linestyle=(0,(2,3)))
ax3.set_xlabel("false-flag rate",fontsize=7.0,color=BK)
ax3.set_ylabel("detection rate",fontsize=7.0,color=BK)
ax3.set_title("operating characteristic",fontsize=7.2,color="#0F2545",fontweight="bold",pad=6)
ax3.legend(fontsize=6.4,frameon=False,loc="lower right",labelcolor=BK)

for dom,col,ls in [("RLV",INK,"-"),("Healthcare",ACC,"--")]:
    s=et[et.domain==dom].sort_values("eta")
    ax4.plot(s.eta,s.attribution_rate,color=col,linestyle=ls,linewidth=1.4)
    ax4.plot(s.eta,s.misattribution_rate,color=col,linestyle=(0,(1,1.6)),linewidth=1.4)
ax4.axvline(0.010,color=BK,linewidth=0.9,linestyle=(0,(3,2)))
ax4.text(0.0115,0.55,"declared",fontsize=6.2,color=BK)
ax4.text(0.034,0.99,"attribution",fontsize=6.4,color=BK,va="top")
ax4.text(0.030,0.22,"misattribution",fontsize=6.4,color=BK,va="bottom")
ax4.set_xlabel("separation margin",fontsize=7.0,color=BK)
ax4.set_ylabel("rate",fontsize=7.0,color=BK)
ax4.set_title("why a margin is needed",fontsize=7.2,color="#0F2545",fontweight="bold",pad=6)

for a in (ax1,ax2,ax3,ax4):
    a.tick_params(labelsize=6.4,colors=BK)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    a.spines['left'].set_color(BK); a.spines['bottom'].set_color(BK)
fig.text(0.5,0.045,"solid lines are the red-light violation configuration, dashed the healthcare vignette; declared operating points are marked in every panel",
         ha="center",fontsize=6.4,color=BK,style="italic")
fig.text(0.012,0.965,"(b)",fontsize=8.2,color="#000000",fontweight="bold",va="top")
fig.savefig("Figure_14.png",facecolor="white")
print("ok")
