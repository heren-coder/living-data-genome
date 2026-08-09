import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

MM=1/25.4
W,H=184.6,120.0
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")
INK="#1F3864"; SUB="#1A1A1A"; ACC="#C55A11"; GRN="#2E7D32"; FRAME="#85B7EB"

def phase(x0,x1,y0,y1,label):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle="round,pad=0,rounding_size=2.0",linewidth=0.9,
        edgecolor=FRAME,facecolor="none",zorder=1))
    ax.text((x0+x1)/2,y1-5.0,label,ha="center",va="center",fontsize=7.6,
            color=INK,fontweight="bold")

def box(x0,x1,y0,y1,title,sub=None,ec=INK,ts=7.2,ss=6.3,r=1.4):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle=f"round,pad=0,rounding_size={r}",linewidth=0.95,
        edgecolor=ec,facecolor="white",zorder=2))
    cx,cy=(x0+x1)/2,(y0+y1)/2
    if sub:
        ns=sub.count("\n")+1
        ax.text(cx,cy+1.2*ns+1.0,title,ha="center",va="center",fontsize=ts,color=ec,
                fontweight="bold",zorder=3)
        ax.text(cx,cy-1.6-0.2*ns,sub,ha="center",va="center",fontsize=ss,color=SUB,
                zorder=3,linespacing=1.28)
    else:
        ax.text(cx,cy,title,ha="center",va="center",fontsize=ts,color=ec,
                fontweight="bold",zorder=3)

def arrow(x0,y0,x1,y1,c=INK,ls="-",lw=1.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",linewidth=lw,
        linestyle=ls,color=c,mutation_scale=7,zorder=5,shrinkA=0,shrinkB=0))

P1=(3,60); P2=(63.5,121); P3=(124.5,181.6)
TOP,BOT=101.0,28.0
# ---- trigger, sitting directly on the phase row ----
box(66,119,107.0,115.0,"Post-incident release",ts=7.4,r=1.2)
arrow(92.5,107.0,92.5,TOP)

phase(*P1,BOT,TOP,"Phase 1 · Commitment")
phase(*P2,BOT,TOP,"Phase 2 · Multi-party governance")
phase(*P3,BOT,TOP,"Phase 3 · Resolution")
arrow(60,65,63.5,65); arrow(121,65,124.5,65)

# ---- Phase 1 ----
box(7,56,78,90,"Governed package","claim · artifact\nexplanation · lineage")
box(7,56,59,71,"Canonical commitment","hash · sign · Merkle anchor\nwrites: accept")
arrow(31.5,78,31.5,71)
ax.text(31.5,48.0,"the package is committed before it is\nshown to any other party",
        ha="center",va="center",fontsize=6.3,color=SUB,style="italic",linespacing=1.3)


def institution(cx,cy,w=6.4,c=INK):
    """Classical-building glyph, matching the receiving institution of Figure 6."""
    h=w*0.62
    ax.add_patch(plt.Polygon([[cx-w/2,cy+h*0.18],[cx,cy+h*0.52],[cx+w/2,cy+h*0.18]],
                 closed=True,fill=False,edgecolor=c,linewidth=0.8,zorder=6))
    ax.add_line(Line2D([cx-w/2,cx+w/2],[cy+h*0.18,cy+h*0.18],color=c,linewidth=0.8,zorder=6))
    for k in range(4):
        x=cx-w*0.34+k*(w*0.68/3)
        ax.add_line(Line2D([x,x],[cy-h*0.30,cy+h*0.12],color=c,linewidth=0.7,zorder=6))
    ax.add_line(Line2D([cx-w*0.46,cx+w*0.46],[cy-h*0.30,cy-h*0.30],color=c,linewidth=0.9,zorder=6))
    ax.add_line(Line2D([cx-w*0.52,cx+w*0.52],[cy-h*0.42,cy-h*0.42],color=c,linewidth=0.9,zorder=6))

# ---- Phase 2 ----
ax.text(92.5,90.6,"Independent institutional signers",ha="center",va="center",
        fontsize=7.0,color=INK,fontweight="bold")
rows=[("Institution A",82.0),("Institution B",74.5),("Institution N",62.5)]
for lab,y in rows:
    box(76,113,y,y+6.6,"",ts=6.8,r=1.2)
    institution(81.5,y+3.3)
    ax.text(96.0,y+3.3,lab,ha="center",va="center",fontsize=6.8,color=INK,
            fontweight="bold",zorder=7)
ax.text(92.5,70.6,"\u22ee",ha="center",va="center",fontsize=9,color=SUB)
BR=71.5
ax.add_line(Line2D([BR,BR],[rows[0][1]+3.3,rows[2][1]+3.3],color=INK,linewidth=0.95,zorder=4))
for _,y in rows:
    ax.add_line(Line2D([BR,76],[y+3.3,y+3.3],color=INK,linewidth=0.95,zorder=4))
arrow(92.5,62.5,92.5,58.0,ls=(0,(3,2)))
box(66,119,44.5,58.0,"Attestation under a BFT quorum",
    "2f + 1 independent signatures over one digest;\nonly the admissibility call is exchanged\nwrites: certify",ts=7.0,ss=6.0)
arrow(92.5,44.5,92.5,43.0,ls=(0,(3,2)))
box(66,119,33.0,43.0,"Institutional consensus",
    "agreement across institution-facing artifacts",ts=7.0,ss=6.0)

ax.text(92.5,30.6,"contestation and review of ancestry \u00b7 writes: contest",
        ha="center",va="center",fontsize=5.9,color=SUB,style="italic")

# ---- Phase 3 ----
box(128,178,80,89,"Revocation decision","evaluated under the regime then in force",ts=7.0,ss=6.0)
ax.add_line(Line2D([153,153],[80,76.5],color=SUB,linewidth=1.0,zorder=5))
ax.add_line(Line2D([140,166],[76.5,76.5],color=SUB,linewidth=1.0,zorder=5))
arrow(140,76.5,140,72.0,c=SUB); arrow(166,76.5,166,72.0,c=ACC)
ax.text(137.6,74.6,"0",ha="right",va="center",fontsize=6.2,color=SUB,fontweight="bold")
ax.text(168.4,74.6,"1",ha="left",va="center",fontsize=6.2,color=ACC,fontweight="bold")
box(128,152,60.0,72.0,"Retain","the artifact remains\nvalid; no lineage\nevent is recorded",ts=6.8,ss=5.8)
box(155,178,60.0,72.0,"Revoke","apoptosis: invalidated,\nhistory preserved\nwrites: revoke",ec=ACC,ts=6.8,ss=5.8)
arrow(166.5,60.0,166.5,55.0,c=SUB,ls=(0,(3,2)))
box(128,178,43.0,55.0,"Regenerated successor","corrected and re-expressed, linked to the\npreserved lineage \u00b7 writes: regenerate",ec=GRN,ts=7.0,ss=6.0)
ax.text(153,36.0,"the successor re-enters through the same\ngate that governed the original release",
        ha="center",va="center",fontsize=6.3,color=SUB,style="italic",linespacing=1.3)

# ---- lineage strip ----
ax.add_patch(FancyBboxPatch((3,3),178.6,20,boxstyle="round,pad=0,rounding_size=2.0",
             linewidth=0.9,edgecolor=FRAME,facecolor="none",zorder=1))
ax.text(92.5,19.4,"Append-only signed-event lineage \u00b7 governance alphabet \u039b, extended and never overwritten",
        ha="center",va="center",fontsize=6.8,color=INK,fontweight="bold")
ev=[("accept","signed",INK),("certify","federated",INK),("contest","challenge",INK),
    ("revoke","invalidate",ACC),("regenerate","successor",GRN)]
n=len(ev); xs=[26.0+i*35.0 for i in range(n)]
for (a,b,c),x in zip(ev,xs):
    ax.add_patch(Circle((x,12.4),2.7,fill=False,edgecolor=c,linewidth=1.0,zorder=3))
    ax.text(x,12.4,"\u2113",ha="center",va="center",fontsize=6.4,color=c,zorder=4)
    ax.text(x,7.8,a,ha="center",va="center",fontsize=6.6,color=c,fontweight="bold")
    ax.text(x,4.8,b,ha="center",va="center",fontsize=6.0,color=SUB)
for i in range(n-1):
    arrow(xs[i]+3.4,12.4,xs[i+1]-3.4,12.4,c=SUB)
arrow(92.5,BOT,92.5,23.6,c=SUB,ls=(0,(3,2)))

fig.savefig("Figure_5.png",facecolor="white")
print("ok")
