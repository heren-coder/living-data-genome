import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

MM=1/25.4
W,H=184.6,136.0
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")

INK="#111111"; SUB="#333333"; ACC="#C55A11"; FRAME="#85B7EB"
C1=(8,47); C2=(53,76); C3=(82,109); C4=(115,145); C5=(151,177)
BOUND=112.0

def frame(y0,y1,title):
    ax.add_patch(FancyBboxPatch((3.0,y0),178.6,y1-y0,
        boxstyle="round,pad=0,rounding_size=2.2",linewidth=0.9,
        edgecolor=FRAME,facecolor="none",zorder=1))
    ax.text(8.0,y1-4.6,title,ha="left",va="center",fontsize=7.6,color=INK,
            fontweight="bold",zorder=3)

def box(x0,x1,y0,y1,title,sub=None,ts=7.4,ss=6.3,ec=INK,r=1.3):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle=f"round,pad=0,rounding_size={r}",linewidth=0.9,
        edgecolor=ec,facecolor="none",zorder=2))
    cx,cy=(x0+x1)/2,(y0+y1)/2
    if sub:
        nt=title.count("\n")+1
        ax.text(cx,cy+1.3*nt+1.4,title,ha="center",va="center",fontsize=ts,color=ec,
                fontweight="bold",zorder=3,linespacing=1.2)
        ax.text(cx,cy-1.6*nt-1.5,sub,ha="center",va="center",fontsize=ss,color=SUB,
                zorder=3,linespacing=1.3)
    else:
        ax.text(cx,cy,title,ha="center",va="center",fontsize=ts,color=ec,
                fontweight="bold",zorder=3,linespacing=1.2)

def arrow(x0,y0,x1,y1,c=INK,ls="-",rad=0.0,lw=0.9):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",linewidth=lw,linestyle=ls,color=c,
        mutation_scale=6,zorder=4,shrinkA=0,shrinkB=0))

def span(x0,x1,y,label,color,tick=1.9):
    ax.add_line(Line2D([x0,x1],[y,y],color=color,linewidth=0.9,zorder=3))
    for x in (x0,x1):
        ax.add_line(Line2D([x,x],[y-tick,y+tick],color=color,linewidth=0.9,zorder=3))
    ax.text((x0+x1)/2,y+3.8,label,ha="center",va="center",fontsize=6.6,
            color=color,style="italic",zorder=3)

# ===================== representation lane =====================
RF0,RF1=99.0,134.0
frame(RF0,RF1,"Representation pathway")
ax.text(177.0,RF1-4.6,"internal throughout",ha="right",va="center",fontsize=6.4,
        color=SUB,style="italic")
UY0,UY1=103.0,123.0
box(*C1,UY0,UY1,"Admissible traces","signal timing log, vehicle trace,\nenvironmental indicators,\npolicy threshold",ss=6.2)
box(*C2,UY0,UY1,"Proxy \u03a0","auditable,\nno prediction")
box(*C3,UY0,UY1,"","")
ax.text((C3[0]+C3[1])/2,UY1-4.4,"Gene state",ha="center",va="center",fontsize=7.4,
        color=INK,fontweight="bold")
for i,g in enumerate(["S","A","D","E"]):
    x=C3[0]+5.4+i*5.6
    ax.add_patch(Circle((x,UY0+8.4),2.1,fill=False,edgecolor=INK,linewidth=0.8,zorder=3))
    ax.text(x,UY0+8.4,g,ha="center",va="center",fontsize=6.0,color=INK,zorder=4)
    ax.add_line(Line2D([x-2.3,x+2.3],[UY0+3.6,UY0+3.6],color="#9A9A9A",linewidth=0.8))
    ax.add_line(Line2D([x-2.3,x+1.4-i*0.9],[UY0+3.6,UY0+3.6],color=INK,linewidth=1.9))
ax.text((C3[0]+C3[1])/2,UY0-2.6,"four coordinates, each with its coverage",
        ha="center",va="center",fontsize=6.0,color=SUB)
box(*C4,UY0,UY1,"Gene feasibility","membership in the\nadmissible region")
for a,b in [(C1[1],C2[0]),(C2[1],C3[0]),(C3[1],C4[0])]:
    arrow(a,(UY0+UY1)/2,b,(UY0+UY1)/2)

# ===================== phase spans =====================
SPY=92.0
span(8.0,BOUND-2.0,SPY,"internal phase, evidence held locally",SUB)
span(BOUND+2.0,177.0,SPY,"institution-facing phase",ACC)

# ===================== release lane =====================
LF0,LF1=34.0,86.0
frame(LF0,LF1,"Release pathway")
LY0,LY1=55.0,75.0
box(*C1,LY0,LY1,"Internal evidence state","reconstruct \u00b7 branch \u00b7 revise\nrevisable, kept local")
box(*C2,LY0,LY1,"Expression \u03a6","candidate\nformation")
box(*C3,LY0,LY1,"Candidate\nartifact","before gating")
box(*C4,LY0,LY1,"Admissibility gate","feasibility, causality,\nprivacy, policy",ec=ACC)
box(*C5,LY0,LY1,"","")
ax.text((C5[0]+C5[1])/2,LY1-4.0,"Released evidence\npackage",ha="center",va="center",
        fontsize=6.6,color=INK,fontweight="bold",linespacing=1.15)
for i,t in enumerate(["claim","artifact","reasons","lineage"]):
    col,row=i%2,i//2
    x0=C5[0]+1.5+col*12.2; y0=LY0+6.2-row*5.6
    ax.add_patch(Rectangle((x0,y0),11.3,4.8,fill=False,edgecolor="#8A8A8A",linewidth=0.7,zorder=3))
    ax.text(x0+5.65,y0+2.4,t,ha="center",va="center",fontsize=5.8,color=SUB,zorder=4)
for a,b in [(C1[1],C2[0]),(C2[1],C3[0]),(C3[1],C4[0]),(C4[1],C5[0])]:
    arrow(a,(LY0+LY1)/2,b,(LY0+LY1)/2)
ax.text((C4[1]+C5[0])/2,(LY0+LY1)/2+3.2,"pass",ha="center",va="center",
        fontsize=6.0,color=SUB,style="italic")

FY=47.0
cx4=(C4[0]+C4[1])/2; cx1=(C1[0]+C1[1])/2
ax.add_line(Line2D([cx4,cx4],[LY0,FY],color=INK,linewidth=0.9,linestyle=(0,(3,2)),zorder=4))
ax.add_line(Line2D([cx1,cx4],[FY,FY],color=INK,linewidth=0.9,linestyle=(0,(3,2)),zorder=4))
arrow(cx1,FY,cx1,LY0,c=INK,ls=(0,(3,2)))
ax.text((cx1+cx4)/2,FY-3.2,"fail: not released, repaired or replaced",ha="center",
        va="center",fontsize=6.4,color=SUB,style="italic")

# interface condition, drawn in ink so that orange stays the boundary colour
arrow(cx4,LY1,cx4,UY0,c=INK,ls=(0,(2,2)),lw=1.0)
ax.text(BOUND-6.0,81.0,"every released artifact admits a gene-level witness;\nthe converse does not hold",
        ha="right",va="center",fontsize=6.4,color=INK,style="italic",linespacing=1.35)

# ===================== externalization boundary =====================
ax.add_line(Line2D([BOUND,BOUND],[3.0,SPY+2.5],color=ACC,linewidth=1.2,
            linestyle=(0,(5,3)),zorder=1))
ax.text(BOUND+2.5,31.6,"externalization boundary",ha="left",va="center",
        fontsize=6.4,color=ACC,style="italic")

# ===================== lineage lane =====================
GF0,GF1=1.5,29.0
frame(GF0,GF1,"Append-only lineage")
R1Y0,R1Y1=16.0,24.0
R2Y0,R2Y1=3.0,11.0
ev1=["accept","certify","contest"]; ev2=["revoke","regenerate"]
x0s=BOUND+3.0; x1s=178.5; gap=2.6
wbox=(x1s-x0s-gap*2)/3
xs1=[]
for i,e in enumerate(ev1):
    a=x0s+i*(wbox+gap); b=a+wbox; xs1.append((a,b))
    box(a,b,R1Y0,R1Y1,e,ts=6.6,r=1.0)
for i in range(2):
    arrow(xs1[i][1],(R1Y0+R1Y1)/2,xs1[i+1][0],(R1Y0+R1Y1)/2)
xs2=[]
for i,e in enumerate(ev2):
    a=x0s+i*(wbox+gap); b=a+wbox; xs2.append((a,b))
    box(a,b,R2Y0,R2Y1,e,ts=6.6,r=1.0)
arrow(xs2[0][1],(R2Y0+R2Y1)/2,xs2[1][0],(R2Y0+R2Y1)/2)
cxa=(xs1[2][0]+xs1[2][1])/2; cxb=(xs2[0][0]+xs2[0][1])/2; MY=13.5
ax.add_line(Line2D([cxa,cxa],[R1Y0,MY],color=INK,linewidth=0.9,zorder=4))
ax.add_line(Line2D([cxb,cxa],[MY,MY],color=INK,linewidth=0.9,zorder=4))
arrow(cxb,MY,cxb,R2Y1)
ax.text(BOUND-4.0,17.0,"extended, never overwritten",ha="right",va="center",
        fontsize=6.4,color=SUB,style="italic")
arrow((C5[0]+C5[1])/2,LY0,(C5[0]+C5[1])/2,R1Y1+0.8,c=INK,ls=(0,(3,2)))

fig.savefig("Figure_2.png",facecolor="white")
print("ok")
