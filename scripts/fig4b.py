import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch
MM=1/25.4
W,H=184.6,80.0
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")
INK="#111111"; SUB="#333333"; ACC="#C55A11"; FRAME="#85B7EB"

def panel(x0,x1,y0,y1,title,sub):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle="round,pad=0,rounding_size=2.0",linewidth=0.9,
        edgecolor=FRAME,facecolor="none",zorder=1))
    ax.text(x0+4.0,y1-4.8,title,ha="left",va="center",fontsize=7.6,color=INK,fontweight="bold")
    ax.text(x0+4.0,y1-9.4,sub,ha="left",va="center",fontsize=6.4,color=SUB,style="italic")

def arrow(p,q,c=INK,lw=1.1):
    ax.add_patch(FancyArrowPatch(tuple(p),tuple(q),arrowstyle="-|>",linewidth=lw,
        color=c,mutation_scale=7,zorder=6,shrinkA=0,shrinkB=0))

def node(p,lab,dx,dy,fs=6.3):
    ax.add_patch(Circle(p,1.4,facecolor="white",edgecolor=INK,linewidth=0.9,zorder=7))
    ax.add_patch(Circle(p,0.55,facecolor=INK,edgecolor="none",zorder=8))
    ax.text(p[0]+dx,p[1]+dy,lab,ha="center",va="center",fontsize=fs,color=INK)

def scene(cx,cy,exact):
    ax.add_patch(Rectangle((cx-26,cy-17),52,34,fill=False,edgecolor=SUB,
                 linewidth=0.9,linestyle=(0,(5,3)),zorder=2))
    ax.add_patch(Rectangle((cx-16,cy-12),32,26,fill=False,edgecolor=ACC,linewidth=1.1,zorder=3))
    ax.text(cx-26,cy+19.4,"G(C)",ha="left",va="center",fontsize=6.6,color=SUB,style="italic")
    ax.text(cx-15.0,cy+15.4,"G(C⁺)",ha="left",va="center",fontsize=6.6,color=ACC,style="italic")
    c=np.array([cx,cy],float)
    if exact:
        g0=np.array([cx-9.0,cy-4.0]); g1=np.array([cx+20.5,cy+8.0])
        g2=np.array([cx+14.5,cy+8.0])
    else:
        g0=np.array([cx+15.0,cy+7.0]); g1=np.array([cx+19.5,cy+9.5])
        g2=g1+0.30*(c-g1)
    node(g0,"genesis",0.0,-4.0); node(g1,"mutation",-1.0,4.8); node(g2,"repair",-10.5,2.4)
    arrow(g0,g1,INK,1.0); arrow(g1,g2,ACC,1.2)
    dm=float(np.linalg.norm(g1-g0)); dr=float(np.linalg.norm(g2-g1))
    return dm,dr

panel(3,91,3,77,"Exact projection","Proposition P1: repair cannot exceed mutation")
dm,dr=scene(47,44,True)
ax.text(47,20.0,f"mutation step {dm:.1f}   ·   repair step {dr:.1f}",ha="center",va="center",
        fontsize=6.6,color=INK,fontweight="bold")
ax.text(47,12.6,"the projection is the nearest admissible point and genesis is already\n"
        "admissible, so the corrective step is at most as long as the mutation step",
        ha="center",va="center",fontsize=6.2,color=SUB,linespacing=1.35)

panel(94,181.6,3,77,"Bounded corrective step","Proposition P2\u2032: the ordering becomes a condition")
# magnified detail of the corner region
bx,by=132.0,40.0; K=2.6
ax.plot([bx-22,bx+26],[by+16,by+16],color=SUB,lw=0.9,ls=(0,(5,3)),zorder=2)
ax.plot([bx+26,bx+26],[by+16,by-14],color=SUB,lw=0.9,ls=(0,(5,3)),zorder=2)
ax.plot([bx-22,bx+12],[by+4,by+4],color=ACC,lw=1.1,zorder=3)
ax.plot([bx+12,bx+12],[by+4,by-14],color=ACC,lw=1.1,zorder=3)
ax.text(bx-21,by+18.6,"G(C)",ha="left",va="center",fontsize=6.6,color=SUB,style="italic")
ax.text(bx-21,by+6.6,"G(C\u207a)",ha="left",va="center",fontsize=6.6,color=ACC,style="italic")
ax.text(bx+26,by-16.4,"detail of the corner region",ha="right",va="center",fontsize=6.0,
        color=SUB,style="italic")
import numpy as _np
g0=_np.array([bx+8.5,by+0.5]); g1=_np.array([bx+10.5,by+8.5])
c=_np.array([bx-14.0,by-9.0])
g2=g1+0.30*(c-g1)
node(g0,"genesis",4.8,-2.6); node(g1,"mutation",8.6,1.6); node(g2,"repair",-1.2,4.0)
arrow(g0,g1,INK,1.0); arrow(g1,g2,ACC,1.3)
dm2=float(_np.linalg.norm(g1-g0))/K; dr2=float(_np.linalg.norm(g2-g1))/K
ax.text(138,20.0,"mutation step shorter than the repair step",ha="center",va="center",
        fontsize=6.6,color=ACC,fontweight="bold")
ax.text(137.8,12.0,"the implemented operator pulls a fixed fraction toward the centre,\n"
        "so a short mutation can be followed by a longer correction,\n"
        "as it is for roughly one candidate in six",
        ha="center",va="center",fontsize=6.2,color=SUB,linespacing=1.35)
fig.savefig("Figure_4.png",facecolor="white")
print("ok")
