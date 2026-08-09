import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
MM=1/25.4
W,H=184.6,58.0
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")
INK="#1F3864"; SUB="#1A1A1A"; ACC="#C55A11"; FRAME="#85B7EB"

def box(x0,x1,y0,y1,t,s=None,ec=INK,ts=6.8,ss=6.0,r=1.3):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=0.95,edgecolor=ec,facecolor="white",zorder=2))
    cx,cy=(x0+x1)/2,(y0+y1)/2
    if s:
        ns=s.count("\n")+1
        ax.text(cx,cy+1.2*ns+0.9,t,ha="center",va="center",fontsize=ts,color=ec,
                fontweight="bold",zorder=3,linespacing=1.2)
        ax.text(cx,cy-1.5-0.25*ns,s,ha="center",va="center",fontsize=ss,color=SUB,
                zorder=3,linespacing=1.28)
    else:
        ax.text(cx,cy,t,ha="center",va="center",fontsize=ts,color=ec,fontweight="bold",
                zorder=3,linespacing=1.2)
def arrow(x0,y0,x1,y1,c=INK,ls="-",lw=1.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",linewidth=lw,linestyle=ls,
                 color=c,mutation_scale=7,zorder=5,shrinkA=0,shrinkB=0))

BOUND=118.0
ax.add_patch(FancyBboxPatch((3,3),178.6,52,boxstyle="round,pad=0,rounding_size=2.0",
             linewidth=0.9,edgecolor=FRAME,facecolor="none",zorder=1))

box(8,48,20,36,"The vehicle's own\nsensor record","retained locally,\nnever exported")
arrow(48,32,55,32); arrow(48,24,55,24)
box(55,113,29,41,"Supervisory use","driver feedback and autonomous correction;\nno artifact is produced",ts=6.6,ss=5.8)
box(55,113,13,25,"Escalation","only when an institution must be addressed",ts=6.6,ss=5.8,ec=ACC)
ax.add_line(Line2D([BOUND,BOUND],[8,49],color=ACC,linewidth=1.2,linestyle=(0,(5,3)),zorder=1))
ax.text(BOUND,5.6,"externalization boundary",ha="center",va="center",fontsize=6.0,
        color=ACC,style="italic")
arrow(113,19,BOUND-1.5,19,c=ACC,ls=(0,(3,2)))
arrow(BOUND+1.5,19,124,19,c=ACC,ls=(0,(3,2)))
box(124,177,13,25,"Gene coordinate and\nreleased package","the only objects that cross",ts=6.6,ss=5.8,ec=ACC)
ax.text(84,47.0,"internal to the node",ha="center",va="center",fontsize=6.2,color=SUB,style="italic")
ax.text(150.5,47.0,"institution-facing",ha="center",va="center",fontsize=6.2,color=ACC,style="italic")
ax.text(84,9.0,"the raw record stays behind in both fates",ha="center",va="center",
        fontsize=6.0,color=SUB,style="italic")

def institution(cx,cy,w=13.0,c=INK):
    h=w*0.62
    # pediment
    ax.add_patch(plt.Polygon([[cx-w/2,cy+h*0.18],[cx,cy+h*0.52],[cx+w/2,cy+h*0.18]],
                 closed=True,fill=False,edgecolor=c,linewidth=1.0,zorder=4))
    ax.add_line(Line2D([cx-w/2,cx+w/2],[cy+h*0.18,cy+h*0.18],color=c,linewidth=1.0,zorder=4))
    # columns
    for k in range(4):
        x=cx-w*0.34+k*(w*0.68/3)
        ax.add_line(Line2D([x,x],[cy-h*0.30,cy+h*0.12],color=c,linewidth=0.9,zorder=4))
    # base
    ax.add_line(Line2D([cx-w*0.46,cx+w*0.46],[cy-h*0.30,cy-h*0.30],color=c,linewidth=1.1,zorder=4))
    ax.add_line(Line2D([cx-w*0.52,cx+w*0.52],[cy-h*0.42,cy-h*0.42],color=c,linewidth=1.1,zorder=4))

institution(146.0,36.0,13.0,ACC)
ax.text(155.0,36.0,"the receiving\ninstitution",ha="left",va="center",fontsize=6.0,
        color=ACC,style="italic",linespacing=1.3)
arrow(146.0,25.0,146.0,30.6,c=ACC,ls=(0,(3,2)))

fig.savefig("Figure_6c.png",facecolor="white")
print("ok")
