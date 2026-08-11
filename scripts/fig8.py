import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
FIGDIR = _os.path.join(_HERE, "..", "figures") + _os.sep
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, sys
import matplotlib.pyplot as plt
from generator import RLV_CONFIG as R, HEALTHCARE_CONFIG as H
sc=pd.read_csv(D+"scenarios.csv")
G=['g_S','g_A','g_D','g_E']; K=['S','A','D','E']
INK="#1F3864"; ACC="#C55A11"; SUB="#1A1A1A"; GRN="#2E7D32"; FRAME="#85B7EB"

def dist(row,cfg):
    b=cfg.feasibility_bounds(row.context_label); g=[row.g_S,row.g_A,row.g_D,row.g_E]
    out=[]
    for j,k in enumerate(K):
        lo,hi=b[k]; hw=(hi-lo)/2
        out.append(max(0.0, lo-g[j], g[j]-hi)/hw)
    return max(out)

d=sc[sc.domain=="RLV"].copy()
d["dist"]=[dist(r,R) for r in d.itertuples()]
piv=d.pivot_table(index=["event_id","candidate_id"],columns="stage_t",values="dist")
# pick an event whose family shows both the dip and one candidate still outside after repair
cand=[e for e,g in piv.groupby(level=0)
      if len(g)>=5 and (g[1]>0).sum()>=2 and (g[2]>0).sum()>=1 and (g[0]==0).sum()>=3]
ev=cand[0]
fam=piv.loc[ev]

MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,86.0*MM),dpi=600)
ax1=fig.add_axes([0.075,0.590,0.9,0.330])
ax2=fig.add_axes([0.075,0.120,0.9,0.310])

xs=[0,1,2]
nf=len(fam); off=np.linspace(-0.13,0.13,nf)
for (idx,row),dx in zip(fam.iterrows(),off):
    xj=[t+dx for t in xs]; y=[row[0],row[1],row[2]]
    ax1.plot(xj,y,color="#BEBEBE",linewidth=0.7,zorder=1)
    for t,v in zip(xj,y):
        ax1.plot([t],[v],marker="o",markersize=4.0,
                 color=(GRN if v<=1e-9 else ACC),zorder=3)
ax1.axhline(0,color=SUB,linewidth=0.9,linestyle=(0,(5,3)))
ax1.text(2.33,0.012,"boundary of G(C)",ha="right",va="bottom",fontsize=6.4,color=SUB)
worst=fam[2].idxmax()
ax1.annotate("still outside after repair",xy=(2,fam.loc[worst,2]),
             xytext=(1.30,fam.loc[worst,2]+0.13),fontsize=6.4,color=ACC,
             arrowprops=dict(arrowstyle="-|>",color=ACC,lw=0.8,mutation_scale=6))
ax1.set_xticks(xs); ax1.set_xticklabels(["Genesis (t = 0)","Mutation (t = 1)","Repair (t = 2)"],fontsize=7.0)
ax1.set_ylabel("normalised distance\nto the admissible region",fontsize=7.0,linespacing=1.3)
ax1.tick_params(axis='y',labelsize=6.6); ax1.set_xlim(-0.35,2.42); ax1.set_ylim(-0.02,0.34)
for s in ("top","right"): ax1.spines[s].set_visible(False)
ax1.spines['left'].set_color(SUB); ax1.spines['bottom'].set_color(SUB)
ax1.set_title("One event: candidates leave the admissible region under mutation and are pulled back by repair",
              fontsize=7.4,color=INK,fontweight="bold",pad=6)
ax1.plot([],[],marker="o",color=GRN,linestyle="none",markersize=4.2,label="inside G(C)")
ax1.plot([],[],marker="o",color=ACC,linestyle="none",markersize=4.2,label="outside G(C)")
ax1.legend(fontsize=6.4,frameon=False,loc="upper left",handletextpad=0.3)

rates={}
for dom,cfg in [("RLV",R),("Healthcare",H)]:
    dd=sc[sc.domain==dom].copy(); dd["dist"]=[dist(r,cfg) for r in dd.itertuples()]
    rates[dom]=[float((dd[dd.stage_t==t]["dist"]<=1e-9).mean()) for t in (0,1,2)]
for dom,col,mk in [("RLV",INK,"o"),("Healthcare",ACC,"s")]:
    ax2.plot(xs,rates[dom],color=col,marker=mk,markersize=4.6,linewidth=1.3,label=dom)
    for t,v in zip(xs,rates[dom]):
        ax2.annotate(f"{v:.3f}",(t,v),textcoords="offset points",
                     xytext=(0,9 if dom=="RLV" else -13),ha="center",fontsize=6.6,color=col)
ax2.set_xticks(xs); ax2.set_xticklabels(["Genesis (t = 0)","Mutation (t = 1)","Repair (t = 2)"],fontsize=7.0)
ax2.set_ylabel("admissibility rate",fontsize=7.0)
ax2.set_ylim(0.50,0.95); ax2.set_xlim(-0.35,2.35); ax2.tick_params(axis='y',labelsize=6.6)
for s in ("top","right"): ax2.spines[s].set_visible(False)
ax2.spines['left'].set_color(SUB); ax2.spines['bottom'].set_color(SUB)
ax2.legend(fontsize=6.6,frameon=False,loc="lower right")
ax2.set_title("All two hundred events: the dip at mutation and the recovery at repair, in both configurations",
              fontsize=7.4,color=INK,fontweight="bold",pad=6)
fig.savefig(FIGDIR + "Figure_8.png",facecolor="white")
print("olay:",ev,"aile:",len(fam))
print("oranlar:",{k:[round(x,3) for x in v] for k,v in rates.items()})
