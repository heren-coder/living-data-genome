import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv("../data/validity_arms.csv")

INK="#1F3864"; ACC="#C55A11"; RISE="#2E7D32"; GREY="#9A9A9A"; BK="#000000"; TIT="#0F2545"

# rows top to bottom, ordered by effect on the aggregate
ARMS=[("enforced admissibility","admissibility enforced,\nnot measured",RISE),
      ("__flat__","three transformations",INK),
      ("instability","instability",ACC),
      ("lineage","lineage breakdown",ACC),
      ("misalignment","semantic misalignment",ACC),
      ("releasability","loss of releasability",ACC)]

MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,78*MM),dpi=600)
axes=[fig.add_axes([0.235,0.245,0.265,0.520]),fig.add_axes([0.640,0.245,0.265,0.520])]

for ax,dom in zip(axes,["RLV","Healthcare"]):
    s=d[d.domain==dom].set_index("arm")
    base=float(s.loc["baseline","Rel"])
    ys=np.arange(len(ARMS))[::-1]
    ax.axvline(0,color=GREY,linewidth=0.8,zorder=1)
    for y,(arm,lab,col) in zip(ys,ARMS):
        dv = 0.0 if arm=="__flat__" else float(s.loc[arm,"Rel"])-base
        ax.plot([0,dv],[y,y],color=col,linewidth=1.1,alpha=0.55,zorder=2,
                solid_capstyle="butt")
        ax.scatter([dv],[y],s=64,marker="o",facecolors=col,edgecolors=col,
                   linewidths=1.4,zorder=4)
        ha,ox = ("right",-10) if dv<0 else ("left",10)
        ax.annotate(("0.000" if dv==0 else f"{dv:+.3f}"),(dv,y),textcoords="offset points",xytext=(ox,0),
                    ha=ha,va="center",fontsize=6.2,color=col)
    ax.annotate(f"undamaged {base:.3f}",(0,len(ARMS)-0.55),
                textcoords="offset points",xytext=(0,2),
                ha="center",va="bottom",fontsize=6.4,color=BK)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _,lab,_ in ARMS] if dom=="RLV" else [],
                       fontsize=6.5,color=BK,linespacing=1.15)
    ax.set_ylim(-0.7,len(ARMS)-0.25)
    ax.set_xlim(-0.288,0.112)
    ax.set_xticks([-0.20,-0.10,0.00])
    ax.set_xlabel("change in the aggregate",fontsize=7.2,color=BK)
    ax.tick_params(labelsize=6.8,colors=BK); ax.tick_params(axis='y',length=0)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BK)
    ax.set_title(dom,fontsize=7.8,color=TIT,fontweight="bold",pad=10)

from matplotlib.lines import Line2D
import numpy as _np
from PIL import Image as _Im
axL,axR=axes
H=fig.get_window_extent().height
_rows=[axL.transData.transform((0,len(ARMS)-1-i))[1]/H for i in range(len(ARMS))]
fig.text(0.5,0.975,"the aggregate falls under every failure mode it targets, holds exactly under\ntransformations that change nothing, and rises when the measurement is removed",
         ha="center",va="top",fontsize=7.4,color=TIT,fontweight="bold",linespacing=1.4)
fig.text(0.5,0.090,"removing the measurement gains 0.048 on average, four times what hiding a real failure mode gains",
         ha="center",fontsize=7.0,color=RISE,fontweight="bold")
fig.text(0.5,0.032,"means over five seeds against the undamaged pipeline on the same events; the three transformations coincide exactly and are drawn as one point",
         ha="center",fontsize=6.5,color=BK,style="italic")
fig.savefig("_probe.png",facecolor="white")
_a=_np.array(_Im.open("_probe.png").convert("L"))
_HH,_WW=_a.shape
_mid=int(0.575*_WW); _pad=int(0.006*_WW); _axL0=int(0.230*_WW)
def _leader(_x0,_x1,_yf):
    if _x1-_x0>0.02:
        fig.add_artist(Line2D([_x0,_x1],[_yf,_yf],transform=fig.transFigure,
                              color="#A0A0A0",linewidth=0.7,
                              linestyle=(0,(1.4,2.2)),zorder=2))
for _yf in _rows:
    _y=int(round((1-_yf)*_HH))
    _narrow=(_a[max(0,_y-3):_y+4,:]<238).any(axis=0)
    _wide=(_a[max(0,_y-26):_y+27,:]<238).any(axis=0)
    # bridge between the two panels
    _left=_np.nonzero(_narrow[:_mid])[0]; _right=_np.nonzero(_narrow[_mid:])[0]
    _leader((_left.max()+_pad)/_WW if len(_left) else 0.50,
            (_right.min()+_mid-_pad)/_WW if len(_right) else 0.94, _yf)
    # leader from the row label to the first content in the left panel
    _lab=_np.nonzero(_wide[:_axL0])[0]
    _cont=_np.nonzero(_narrow[_axL0:_mid])[0]
    if len(_lab) and len(_cont):
        _leader((_lab.max()+_pad)/_WW,(_cont.min()+_axL0-_pad)/_WW,_yf)
import os as _os; _os.remove("_probe.png")
fig.savefig("Figure_16.png",facecolor="white")
print("ok")
