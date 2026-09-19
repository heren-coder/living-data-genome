import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
lv=pd.read_csv("../data/lime_vs_shapley.csv"); gm=pd.read_csv("../data/shapley_geometry.csv")
INK="#1F3864"; ACC="#C55A11"; BK="#000000"; GRN="#2E7D32"
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,68.8*MM),dpi=600)
Y,H=0.245,0.53
a1=fig.add_axes([0.112,Y,0.205,H])
a2=fig.add_axes([0.447,Y,0.210,H])
a3=fig.add_axes([0.748,Y,0.218,H])

def g(dom,c): return float(lv[lv.domain==dom][c].iloc[0])
names=["counterfactual\nprobe","local surrogate\n(LIME)","interventional\nShapley"]
M=np.full((3,3),np.nan)
M[0,1]=M[1,0]=g("RLV","agree_cf_lime")
M[0,2]=M[2,0]=g("RLV","agree_cf_shapley")
M[1,2]=M[2,1]=g("RLV","agree_shapley_lime")
Mh=np.full((3,3),np.nan)
Mh[0,1]=Mh[1,0]=g("Healthcare","agree_cf_lime")
Mh[0,2]=Mh[2,0]=g("Healthcare","agree_cf_shapley")
Mh[1,2]=Mh[2,1]=g("Healthcare","agree_shapley_lime")
for i in range(3):
    for j in range(3):
        if i==j:
            a1.add_patch(Rectangle((j-0.5,i-0.5),1,1,facecolor="#F2F2F2",edgecolor="white",lw=1.2))
            continue
        v=M[i,j]
        col = GRN if v>0.6 else "#BFBFBF"
        a1.add_patch(Rectangle((j-0.5,i-0.5),1,1,facecolor=col,alpha=0.30 if v<0.6 else 0.28,
                     edgecolor="white",lw=1.2))
        a1.text(j,i-0.12,f"{v:.2f}",ha="center",va="center",fontsize=8.0,
                color=(GRN if v>0.6 else BK),fontweight="bold")
        a1.text(j,i+0.24,f"{Mh[i,j]:.2f}",ha="center",va="center",fontsize=6.6,color=ACC)
a1.set_xlim(-0.5,2.5); a1.set_ylim(2.5,-0.5)
a1.set_xticks(range(3)); a1.set_yticks(range(3))
a1.set_xticklabels(["counterf.","surrogate","Shapley"],fontsize=6.4,color=BK)
a1.set_yticklabels(names,fontsize=6.4,color=BK,linespacing=1.15)
a1.tick_params(length=0,colors=BK)
for s in a1.spines.values(): s.set_visible(False)
a1.set_title("pairwise agreement",fontsize=7.6,
             color="#0F2545",fontweight="bold",pad=12)
a1.set_xlabel("RLV in bold, healthcare below; chance is 0.25",fontsize=6.6,color=BK,style="italic",labelpad=5)

rows=[("local\nreference","binds this package",
       [g("RLV","agree_cf_lime"),g("Healthcare","agree_cf_lime")]),
      ("context-pool\nreference","decisive in this context",
       [float(gm[gm.domain=="RLV"].pool_conc.mean()),
        float(gm[gm.domain=="Healthcare"].pool_conc.mean())])]
y=np.arange(2); h=0.28
for i,(dom,col) in enumerate([("RLV",INK),("Healthcare",ACC)]):
    v=[r[2][i] for r in rows]
    a2.barh(y+(0.5-i)*h,v,height=h,color=col,alpha=0.9,label=dom)
    for yi,vi in zip(y+(0.5-i)*h,v):
        a2.text(vi+0.02,yi,f"{vi:.2f}",va="center",fontsize=7.2,color=col,fontweight="bold")
a2.set_yticks(y); a2.set_yticklabels([r[0] for r in rows],fontsize=7.0,color=BK,linespacing=1.2)
for yi,r in zip(y,rows):
    a2.text(0.03,yi+0.40,r[1],fontsize=6.4,color=BK,style="italic",va="center")
a2.set_xlim(0,1.18); a2.set_ylim(1.75,-0.62)
a2.set_xlabel("consistency at its own level",fontsize=7.2,color=BK,labelpad=3)
a2.tick_params(axis='x',labelsize=6.6,colors=BK); a2.tick_params(axis='y',length=0,colors=BK)
for s in ("top","right","left"): a2.spines[s].set_visible(False)
a2.spines['bottom'].set_color(BK)
a2.legend(fontsize=6.8,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.11),ncol=2,
          handlelength=1.0,columnspacing=1.0,labelcolor=BK)
a2.set_title("each reference, its own question",fontsize=7.6,
             color="#0F2545",fontweight="bold",pad=12)

for dom,col,mk in [("RLV",INK,"o"),("Healthcare",ACC,"s")]:
    s=gm[gm.domain==dom]
    a3.scatter(s.gap,s.pool_conc,s=34,marker=mk,facecolors="none",edgecolors=col,
               linewidths=1.2,label=dom,zorder=3)
w=gm[gm.gap==gm.gap.min()].iloc[0]
a3.annotate("two coordinates equally tight",xy=(w.gap,w.pool_conc),xytext=(0.0035,0.505),
            fontsize=6.4,color=BK,arrowprops=dict(arrowstyle="-|>",color=BK,lw=0.8,mutation_scale=6))
a3.set_xlabel("margin between the two tightest",fontsize=7.2,color=BK,labelpad=3)
a3.set_ylabel("same top attribution",fontsize=7.2,color=BK)
a3.tick_params(labelsize=6.6,colors=BK); a3.set_ylim(0.35,1.05)
for s in ("top","right"): a3.spines[s].set_visible(False)
a3.spines['left'].set_color(BK); a3.spines['bottom'].set_color(BK)
a3.legend(fontsize=6.8,frameon=False,loc="lower right",labelcolor=BK,handletextpad=0.4,borderaxespad=0.3,handlelength=1.2)
a3.set_title("Corollary C2: geometry fixes it",fontsize=7.6,color="#0F2545",fontweight="bold",pad=12)
fig.savefig("Figure_10.png",facecolor="white")
print("ok")
