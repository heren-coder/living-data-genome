import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, sys
import matplotlib.pyplot as plt
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from generator import RLV_CONFIG as R, HEALTHCARE_CONFIG as H
from rel_computation import linear_cka, RNG_SEED
D="../data/"
sc=pd.read_csv(D+"scenarios.csv")
G=['g_S','g_A','g_D','g_E']
INK="#1F3864"; ACC="#C55A11"; GRN="#2E7D32"; SUB="#1A1A1A"
MM=1/25.4
fig=plt.figure(figsize=(184.6*MM,73.1*MM),dpi=600)
axes=[fig.add_axes([0.075,0.215,0.255,0.520]),
      fig.add_axes([0.385,0.215,0.255,0.520]),
      fig.add_axes([0.735,0.215,0.245,0.560])]
CKA={}
for ax,(dom,cfg) in zip(axes[:2],[("RLV",R),("Healthcare",H)]):
    d=sc[sc.domain==dom]
    a=d[d.stage_t==0].sort_values(["event_id","candidate_id"])
    b=d[d.stage_t==2].sort_values(["event_id","candidate_id"])
    gen=a[G].values; rep=b[G].values
    ev=a.event_id.values; ctx={e:c for e,c in zip(d.event_id,d.context_label)}
    rng=np.random.default_rng(RNG_SEED+7)
    cf_by={e:np.clip(cfg.context_means[ctx[e]]+rng.normal(0,0.16,size=4),0,1) for e in np.unique(ev)}
    cf=np.array([cf_by[e] for e in ev])
    CKA[dom]=(linear_cka(gen,rep),linear_cka(gen,cf),linear_cka(rep,cf))
    X=np.vstack([gen,rep,cf]); Xc=X-X.mean(0)
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    P=Xc@Vt[:2].T; var=(S**2/(S**2).sum())[:2]
    n=len(gen)
    cats=list(cfg.context_categories); cmap={c:i for i,c in enumerate(cats)}
    cols=["#1F3864","#C55A11","#2E7D32","#7B3FA0"]
    ci=np.array([cmap[ctx[e]] for e in ev])
    for k,(mk,lab,sz) in enumerate([("o","genesis",7),("^","repair",7),("x","case-facing",9)]):
        seg=P[k*n:(k+1)*n]
        for j,c in enumerate(cats):
            m=ci==j
            ax.scatter(seg[m,0],seg[m,1],s=sz,marker=mk,linewidths=0.45,
                       facecolors=("none" if mk!="x" else cols[j]),edgecolors=cols[j],alpha=0.72)
    ax.set_title(dom,fontsize=7.8,color="#0F2545",fontweight="bold",pad=5)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)",fontsize=7.2,color="#000000")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)",fontsize=7.2,color="#000000")
    ax.tick_params(labelsize=6.8,colors="#000000")
    for s in ("top","right"): ax.spines[s].set_visible(False)

ax=axes[2]
labels=["genesis\nrepair","genesis\ncase-facing","repair\ncase-facing"]
x=np.arange(3); w=0.36
for i,(dom,col) in enumerate([("RLV",INK),("Healthcare",ACC)]):
    v=CKA[dom]
    ax.bar(x+(i-0.5)*w,v,width=w,color=col,alpha=0.85,label=dom)
    for xi,vi in zip(x+(i-0.5)*w,v):
        ax.text(xi,vi+0.02,f"{vi:.2f}",ha="center",va="bottom",fontsize=6.8,color=col,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6.8,linespacing=1.25,color="#000000")
ax.set_ylim(0,1.22); ax.set_ylabel("pairwise alignment",fontsize=7.2,color="#000000")
ax.tick_params(axis='y',labelsize=6.8,colors='#000000')
ax.legend(fontsize=6.6,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.0),ncol=2,
          handlelength=1.0,columnspacing=1.0,handletextpad=0.4,labelcolor="#000000")
ax.set_title("what the component measures",fontsize=7.8,color="#0F2545",fontweight="bold",pad=5)
for s in ("top","right"): ax.spines[s].set_visible(False)

hs=[plt.Line2D([],[],marker=m,color=SUB,linestyle="none",markersize=4,markerfacecolor="none",
    markeredgewidth=0.6) for m in ["o","^","x"]]
fig.legend(hs,["genesis (t = 0)","repair (t = 2)","case-facing (independent)"],
           loc="lower center",bbox_to_anchor=(0.355,-0.012),ncol=3,fontsize=6.8,frameon=False,
           handletextpad=0.4,columnspacing=2.2,labelcolor="#000000")

fig.text(0.355,0.855,"colour marks the operating context, marker marks the view",
         ha="center",va="center",fontsize=7.8,color="#0F2545",fontweight="bold")
fig.savefig("Figure_9.png",facecolor="white")
print({k:[round(x,3) for x in v] for k,v in CKA.items()})
