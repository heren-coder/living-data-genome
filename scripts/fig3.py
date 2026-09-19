import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D
MM=1/25.4
W,H=184.6,106.0
HFIG=128.0; VS=1.2075
INK="#1F3864"; ACC="#C55A11"; GRN="#2E7D32"; PUR="#7B3FA0"; BK="#000000"; GREY="#8A8A8A"
fig=plt.figure(figsize=(W*MM,HFIG*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x0,x1,y0,y1,ec,lw=1.0,fc="white",r=1.4,z=2):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle=f"round,pad=0,rounding_size={r}",linewidth=lw,edgecolor=ec,facecolor=fc,zorder=z))
def arrow(x0,y0,x1,y1,c=INK,ls="-",lw=0.9,ms=6.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",linewidth=lw,linestyle=ls,
        color=c,mutation_scale=ms,zorder=5,shrinkA=0,shrinkB=0))

ax.text(W/2,103.6,"Internal states evolve and are revised; institution-facing artifacts are externalized and governed.",
        ha="center",va="center",fontsize=7.8,color=BK,style="italic")
BX=93.0
ax.add_line(Line2D([6,BX-9],[98.4,98.4],color=INK,linewidth=1.0))
ax.text(46,99.6,"INTERNAL PHASE",ha="center",fontsize=7.4,color=INK,fontweight="bold")
ax.text(46,96.2,"revisable, not released",ha="center",fontsize=7.1,color=INK,style="italic")
ax.add_line(Line2D([BX+9,179],[98.4,98.4],color=GRN,linewidth=1.0))
ax.text(143,99.6,"EXTERNALIZATION AND GOVERNANCE",ha="center",fontsize=7.4,color=GRN,fontweight="bold")
ax.text(140,96.2,"institution-facing",ha="center",fontsize=7.1,color=GRN,style="italic")
ax.text(BX,99.6,"EXTERNALIZATION",ha="center",fontsize=6.6,color=ACC,fontweight="bold")
ax.text(BX,97.2,"BOUNDARY",ha="center",fontsize=6.6,color=ACC,fontweight="bold")
for x in (6,BX-9,BX+9,179):
    ax.plot([x],[98.4],marker="o",markersize=2.6,color=(ACC if abs(x-BX)<10 else (INK if x<BX else GRN)))

LAY=[(5,43.5,"L1","Generation Layer","internal candidate formation",INK,
      "Dominant object:  E","internal evidentiary state",
      ["Scenario construction","Bounded generation","Feasibility exploration","Internal hypothesis pool"]),
     (46,84,"L2","Representation Layer","gene-level witnessing",ACC,
      "Dominant object:  $z=(g,q)$","measurable gene state",
      ["Encoding and normalization","Consistency screening","Similarity and clustering","Quality and uncertainty"]),
     (102,142,"L3","Externalization Layer","internal to institution-facing",INK,
      "Dominant transition:  $z \\rightarrow R$","boundary transformation",
      ["Expression across the boundary","Artifact-level admissibility","Gene-level feasibility witness",
       "Explanation payload","Policy and constraint compliance"]),
     (143.5,181.5,"L4","Governance Layer","institution-facing",GRN,
      "Dominant object:  R","released artifact and lineage",
      ["Certification and signing","Append-only lineage anchoring","Review and contestation",
       "Revocation and regeneration"])]
TOP,BOT=94.0,57.0
for x0,x1,tag,name,sub,col,dom,domsub,bul in LAY:
    box(x0,x1,BOT,TOP,col,lw=1.3)
    box(x0+1.4,x0+6.4,TOP-5.6,TOP-1.4,col,lw=0,fc=col,r=0.9,z=3)
    ax.text(x0+3.9,TOP-3.5,tag,ha="center",va="center",fontsize=8.3,color="white",fontweight="bold",zorder=4)
    ax.text(x0+7.6,TOP-2.7,name,ha="left",va="center",fontsize=7.0,color=col,fontweight="bold")
    ax.text(x0+7.2,TOP-5.3,sub,ha="left",va="center",fontsize=6.0,color=BK,style="italic")
    ax.text(x0+2.0,TOP-9.4,dom,ha="left",va="center",fontsize=6.8,color=BK,fontweight="bold")
    ax.text(x0+2.0,TOP-12.2,domsub,ha="left",va="center",fontsize=6.4,color=BK,style="italic")
    ax.add_line(Line2D([x0+2.0,x1-2.0],[TOP-14.2,TOP-14.2],color=GREY,linewidth=0.8))
    for i,t in enumerate(bul):
        y=TOP-17.2-i*3.5
        ax.plot([x0+2.3],[y],marker="o",markersize=1.5,color=col)
        ax.text(x0+3.7,y,t,ha="left",va="center",fontsize=6.0,color=BK)

ax.add_line(Line2D([BX,BX],[BOT-1.5,96.0],color=ACC,linewidth=1.2,linestyle=(0,(5,3)),zorder=1))
ax.add_patch(FancyBboxPatch((BX-3.6,76.0),7.2,7.2,boxstyle="round,pad=0,rounding_size=1.2",
             linewidth=1.1,edgecolor=ACC,facecolor="white",zorder=4))
ax.add_patch(Rectangle((BX-1.9,77.6),3.8,2.9,fill=False,edgecolor=ACC,linewidth=1.0,zorder=5))
th=np.linspace(0,np.pi,60)
ax.plot(BX+1.25*np.cos(th),80.5+1.25*np.sin(th),color=ACC,linewidth=1.0,zorder=5)
ax.add_patch(Rectangle((BX-8.6,69.6),17.2,6.2,facecolor="white",edgecolor="none",zorder=3))
ax.text(BX,73.4,"admissibility",ha="center",fontsize=6.4,color=ACC,fontweight="bold",zorder=4)
ax.text(BX,71.0,"gate",ha="center",fontsize=6.4,color=ACC,fontweight="bold",zorder=4)

SY0,SY1=36.0,54.0
box(5,180,SY0,SY1,GREY,lw=1.0)
ax.text(9.5,45.0,"state\nflow",ha="center",va="center",fontsize=7.6,color=BK,fontweight="bold",linespacing=1.2)
ax.add_line(Line2D([15,15],[SY0+1.5,SY1-1.5],color=GREY,linewidth=0.8))
GRP=[(16,52,INK,"internal evidentiary states","revisions allowed, no external release",
      ["$E_{i,t}$","$E_{i,t+1}$","$E_{i,t^*}$"]),
     (53,87.5,ACC,"gene-level states","",
      ["$z_{i,t}$","$z_{i,t+1}$","$z_{i,t^*}$"]),
     (98.5,141,INK,"artifact candidates","gate must be satisfied here",
      ["$R^{*}_{i,t}$","$R^{*}_{i,t+1}$","$R^{*}_{i,t^*}$"]),
     (144,179,GRN,"released artifacts","governed and auditable",
      ["$R_{i,t}$","$R_{i,t+1}$","$R_{i,t^*}$"])]
for x0,x1,col,lab,note,syms in GRP:
    ax.text((x0+x1)/2,52.4,lab,ha="center",fontsize=7.0,color=col,fontweight="bold")
    RX,RY=4.4,3.3/VS
    xs=np.linspace(x0+RX+0.6,x1-RX-0.6,len(syms))
    for k,(x,sy) in enumerate(zip(xs,syms)):
        ax.add_patch(Ellipse((x,46.6),2*RX,2*RY,fill=False,edgecolor=col,linewidth=1.2,zorder=3))
        ax.text(x,46.6,sy,ha="center",va="center",fontsize=6.6,color=col,fontweight="bold",zorder=4)
        if k<len(syms)-1:
            # solid arrow for one revision step, dashed for the elided steps up to t*
            arrow(x+RX+0.5,46.6,xs[k+1]-RX-0.5,46.6,c=col,lw=1.0,ls=("-" if k==0 else (0,(2.2,1.6))))
    if note:
        ax.text((x0+x1)/2,39.0,note,ha="center",fontsize=7.0,color=col,style="italic")
arrow(87.7,46.6,BX-3.9,46.6,c=ACC,ls=(0,(3,2)),lw=1.0)
arrow(BX+3.9,46.6,98.3,46.6,c=ACC,ls=(0,(3,2)),lw=1.0)
arrow(141.4,46.6,143.6,46.6,c=GRN,lw=1.0)

TY0,TY1=17.0,34.0
rows=[("stage index  s",[str(i) for i in range(1,13)]),
      ("stage groups",["S1\u2013S3"]*3+["S4\u2013S6"]*3+["S7\u2013S9"]*3+["S10\u2013S12"]*3),
      ("layer map  L(s)",["L1"]*3+["L2"]*3+["L3"]*3+["L4"]*3)]
lab_w=26.0; cell=(175.0-lab_w)/12
for r,(lab,vals) in enumerate(rows):
    y1=TY1-r*5.6; y0=y1-5.6
    ax.text(7.0,(y0+y1)/2,lab,ha="left",va="center",fontsize=7.3,color=BK,fontweight="bold")
    for c,v in enumerate(vals):
        x=5+lab_w+c*cell
        colr=[INK,ACC,INK,GRN][c//3]
        if r>0 and c%3!=0: 
            ax.add_line(Line2D([x,x+cell],[y0,y0],color=GREY,linewidth=0.4)); continue
        span=cell*(3 if r>0 else 1)
        ax.add_patch(Rectangle((x,y0),span,5.6,fill=False,edgecolor=GREY,linewidth=0.4,zorder=2))
        ax.text(x+span/2,(y0+y1)/2,v,ha="center",va="center",fontsize=7.3,
                color=(colr if r>0 else BK),fontweight=("bold" if r>0 else "normal"))
    if r==0:
        for c in range(12):
            x=5+lab_w+c*cell
            ax.add_patch(Rectangle((x,y0),cell,5.6,fill=False,edgecolor=GREY,linewidth=0.4,zorder=2))

box(5,180,1.5,15.0,GREY,lw=1.0)
ax.text(7.5,12.9,"Notation",ha="left",fontsize=7.6,color=BK,fontweight="bold")
NOTES=[("$E_{i,t}$","internal evidentiary state, revisable"),
       ("$z_{i,t}=(g,q)$","gene-level state, measurable witness"),
       ("$R^{*}_{i,t}$","candidate artifact prior to release"),
       ("$R_{i,t}$","released artifact, governed"),
       ("$s$","stage index, one to twelve"),
       ("$L(s)$","layer assignment map")]
for i2,(sym,txt) in enumerate(NOTES):
    x=7.5+(i2//3)*60.0; y=10.2-(i2%3)*2.9
    ax.text(x,y,sym,ha="left",va="center",fontsize=6.6,color=INK)
    ax.text(x+14.5,y,txt,ha="left",va="center",fontsize=6.4,color=BK)
ax.add_line(Line2D([116.0,116.0],[3.0,13.5],color=GREY,linewidth=0.5))
ax.text(119.0,12.9,"Regime transition rule",ha="left",fontsize=7.6,color=ACC,fontweight="bold")
for k,line in enumerate(["A layer transition is permitted only when",
                         "the decision regime changes, that is when",
                         "$L(s)\\neq L(s{+}1)$ implies $\\Delta reg(s)\\geq 1$"]):
    ax.text(119.0,10.2-k*2.8,line,ha="left",va="center",fontsize=7.3,color=BK)
fig.savefig("Figure_3.png",facecolor="white")
print("ok")
