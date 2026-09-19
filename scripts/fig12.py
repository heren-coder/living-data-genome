import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
HALO=[pe.withStroke(linewidth=2.2,foreground="white")]
from matplotlib.colors import LinearSegmentedColormap
m=pd.read_csv("../data/operating_map.csv")
# Declared agreement boundary: the same statistic the text and Table 6 report,
# i.e. the CV=0.5 crossing interpolated on the 25-seed sweep of kappa_boundary.csv.
# The band previously used the median across tightening columns of the first
# grid point exceeding CV=0.5 on this figure's coarser 5-seed sweep, which is a
# different statistic and gave a different number (17.5 / 20 against 16.3 / 17.0).
_kb=pd.read_csv("../data/kappa_boundary.csv")
def declared_boundary(dom):
    k=_kb[_kb.domain==dom].sort_values("variance_pct")
    cv,v=k.cv.values,k.variance_pct.values
    i=next(j for j in range(len(cv)) if cv[j]>0.5)
    return float(np.interp(0.5,[cv[i-1],cv[i]],[v[i-1],v[i]]))
INK="#1F3864"; ACC="#C55A11"; BK="#000000"
cmap=LinearSegmentedColormap.from_list("g",["#FFFFFF","#CBD9EC","#7FA6D0","#2F5C96","#12305C"])
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,88*MM),dpi=600)
T=sorted(m.tightening.unique()); V=sorted(m.variance_pct.unique())
LAY=[(0.070,0.300),(0.530,0.300)]
im=None
for (x0,wd),dom in zip(LAY,["RLV","Healthcare"]):
    ax  =fig.add_axes([x0,        0.400,wd,      0.400])
    axr =fig.add_axes([x0+wd+0.012,0.400,0.072,  0.400])
    axb =fig.add_axes([x0,        0.215,wd,      0.135])
    s=m[m.domain==dom]
    Z =s.pivot(index="variance_pct",columns="tightening",values="rel_gap_mean").values
    KM=s.pivot(index="variance_pct",columns="tightening",values="kappa_mean").values
    KS=s.pivot(index="variance_pct",columns="tightening",values="kappa_sd").values
    CV=KS/KM
    im=ax.pcolormesh(np.array(T),np.array(V),Z,cmap=cmap,vmin=0.0,vmax=0.55,shading="nearest")
    vb=declared_boundary(dom)
    ax.axhspan(vb,max(V)+1.5,facecolor=ACC,alpha=0.13,zorder=2)
    ax.axhline(vb,color=ACC,linewidth=1.4,linestyle=(0,(5,3)),zorder=3)
    ax.text(T[0]-0.002,vb-1.1,"agreement no longer stable",
            fontsize=6.2,color=ACC,va="top",ha="left",zorder=6,path_effects=HALO)
    peak=[T[int(np.argmax(Z[i]))] for i in range(len(V))]
    ax.plot(peak,V,color="white",linewidth=1.8,zorder=4)
    ax.plot(peak,V,color=BK,linewidth=0.7,zorder=5)
    ax.plot([0.05],[5],marker="o",markersize=6,markerfacecolor="white",
            markeredgecolor=BK,markeredgewidth=1.4,zorder=6)
    ax.annotate("operating point",(0.05,5),textcoords="offset points",xytext=(11,5),
                fontsize=6.4,color=BK,zorder=6,path_effects=HALO)
    ax.set_ylabel("calibration variance (%)" if dom=="RLV" else "",fontsize=7.0,color=BK)
    ax.tick_params(labelsize=6.6,colors=BK,labelbottom=False)
    ax.set_title(dom,fontsize=7.8,color="#0F2545",fontweight="bold",pad=6)

    jt=T.index(0.100)
    axr.plot(Z[:,jt],V,color=INK,linewidth=1.4,marker="o",markersize=2.6)
    axr.set_xlim(0,0.55); axr.set_ylim(ax.get_ylim())
    axr.tick_params(labelsize=6.0,colors=BK,labelleft=False)
    axr.set_xticks([0,0.25,0.5])
    for sp in ("top","right"): axr.spines[sp].set_visible(False)
    rng_v=Z[:,jt].max()-Z[:,jt].min()
    axr.set_title(f"across variance\nspan {rng_v:.3f}",fontsize=6.4,color=INK,pad=4)

    iv=V.index(5)
    axb.plot(T,Z[iv,:],color=ACC,linewidth=1.4,marker="o",markersize=2.6)
    axb.set_xlim(ax.get_xlim())
    axb.tick_params(labelsize=6.6,colors=BK)
    axb.set_xlabel("governance tightening magnitude",fontsize=7.0,color=BK)
    for sp in ("top","right"): axb.spines[sp].set_visible(False)
    rng_t=Z[iv,:].max()-Z[iv,:].min()
    axb.text(0.02,0.94,f"across tightening, span {rng_t:.3f}",transform=axb.transAxes,
             ha="left",va="top",fontsize=6.6,color="#0F2545",fontweight="bold")
    axb.set_ylim(0,0.72)

cax=fig.add_axes([0.930,0.400,0.014,0.400])
cb=fig.colorbar(im,cax=cax); cb.set_label("reliability gained by regeneration over discard",
                                          fontsize=6.6,color=BK)
cb.ax.tick_params(labelsize=6.2,colors=BK)
fig.text(0.5,0.955,"one constant moves the gain and the other does not: the two can be declared and swept independently",
         ha="center",fontsize=7.6,color="#0F2545",fontweight="bold")
fig.text(0.5,0.045,"side strip: the gain along calibration variance at fixed tightening\nlower strip: the gain along tightening at the operating variance",
         ha="center",fontsize=6.5,color=BK,style="italic",linespacing=1.4)
fig.text(0.012,0.965,"(b)",fontsize=8.2,color="#000000",fontweight="bold",va="top")
fig.savefig("Figure_12.png",facecolor="white")
print("ok")
