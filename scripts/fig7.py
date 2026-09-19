"""Schematic source of Figure 7.

Draws the content and layout of the figure: the four columns, the layers and
stages of each phase, the operators and the equations. The published Figure 7
(figures/Figure_7_original_600dpi.png) is an illustrated rendering of this
schematic, drawn by hand with the same labels; it is the one element of the
article, together with the graphical abstract, that the pipeline does not
generate. Output: Figure_7_schematic.png in this directory.
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge, Polygon
from matplotlib.lines import Line2D
MM=1/25.4
W,H=184.6,168.0
INK="#1F3864"; ACC="#C55A11"; GRN="#2E7D32"; PUR="#7B3FA0"; BK="#000000"; GREY="#9A9A9A"; FRAME="#85B7EB"
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x0,x1,y0,y1,ec,lw=0.9,fc="white",r=1.4,z=2,ls="-"):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,edgecolor=ec,facecolor=fc,zorder=z,linestyle=ls))
def arrow(x0,y0,x1,y1,c=INK,ls="-",lw=1.0,ms=7.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",linewidth=lw,linestyle=ls,
        color=c,mutation_scale=ms,zorder=6,shrinkA=0,shrinkB=0))
def wave(x0,x1,y,amp,col,n=260,f=7.0,seed=0):
    rng=np.random.default_rng(seed); x=np.linspace(x0,x1,n)
    y2=y+amp*np.sin((x-x0)*f)+amp*0.45*rng.normal(0,1,n)*0.35
    ax.plot(x,y2,color=col,linewidth=0.55,zorder=4)
def numtag(x,y,n,col):
    ax.add_patch(Circle((x,y),2.6,facecolor=col,edgecolor="none",zorder=5))
    ax.text(x,y,str(n),ha="center",va="center",fontsize=6.6,color="white",fontweight="bold",zorder=6)
def lock(cx,cy,s=1.0,col=INK):
    ax.add_patch(FancyBboxPatch((cx-1.7*s,cy-1.7*s),3.4*s,2.6*s,boxstyle="round,pad=0,rounding_size=0.4",
        linewidth=0.9,edgecolor=col,facecolor="white",zorder=5))
    th=np.linspace(0,np.pi,50)
    ax.plot(cx+1.1*s*np.cos(th),cy+0.9*s+1.1*s*np.sin(th),color=col,linewidth=0.9,zorder=5)

ax.text(W/2,164.6,"From local sensing to reliability assessment without exposing raw data",
        ha="center",va="center",fontsize=8.6,color=BK,fontweight="bold")

# ---------------- panel 1 ----------------
P1=(3,45); box(P1[0],P1[1],62,160,FRAME,lw=0.9)
numtag(P1[0]+4.5,156.5,1,INK)
ax.text(27.5,157.4,"In-Vehicle Traces",ha="center",fontsize=7.0,color=INK,fontweight="bold")
ax.text(26,154.4,"local and protected",ha="center",fontsize=6.2,color=BK,style="italic")
ax.text(24,150.4,"raw multi-modal traces  $u_{i,t}$",ha="center",fontsize=6.6,color=BK,fontweight="bold")
CH=[("brake",145.0,ACC,1),("speed",138.5,INK,2),("signal timing",132.0,GRN,3),
    ("position",125.5,ACC,4),("steering",119.0,PUR,5)]
for lab,y,col,sd in CH:
    ax.text(5.5,y,lab,ha="left",va="center",fontsize=6.0,color=BK)
    ax.add_line(Line2D([20.5,20.5],[y-2.2,y+2.2],color=GREY,linewidth=0.5,linestyle=(0,(2,2))))
    ax.plot([20.5],[y],marker="o",markersize=2.2,color=col)
    wave(21.5,43,y,1.1,col,seed=sd)
ax.text(24,114.0,"environment  ·  traffic density  ·  policy",ha="center",fontsize=6.0,color=BK,style="italic")
# car glyph
cx,cy=24,100.0
ax.add_patch(FancyBboxPatch((cx-11,cy-2.6),22,5.2,boxstyle="round,pad=0,rounding_size=1.6",
             linewidth=1.0,edgecolor=INK,facecolor="white",zorder=3))
ax.add_patch(FancyBboxPatch((cx-5.5,cy+2.0),11,3.4,boxstyle="round,pad=0,rounding_size=1.2",
             linewidth=1.0,edgecolor=INK,facecolor="white",zorder=3))
for wx in (cx-6.5,cx+6.5):
    ax.add_patch(Circle((wx,cy-3.0),1.7,facecolor="white",edgecolor=INK,linewidth=1.0,zorder=4))
th=np.linspace(np.pi*0.15,np.pi*0.85,80)
ax.plot(cx+16*np.cos(th),cy-4.5+5.5*np.sin(th),color=FRAME,linewidth=0.7,linestyle=(0,(3,2)),zorder=2)
lock(cx+13.5,cy-3.0,0.9,INK)
box(5,43,86,93,INK,lw=1.0)
ax.text(24,89.5,"raw traces stay inside the vehicle",ha="center",va="center",
        fontsize=6.6,color=INK,fontweight="bold")

# proxy column
ax.text(48.5,140.0,"$\\Pi$",ha="center",fontsize=11,color=BK)
ax.text(48.5,135.6,"trace $\\rightarrow$ gene",ha="center",fontsize=5.6,color=BK)
for k,g in enumerate("SADE"):
    ax.add_patch(Rectangle((44.4+k*2.1,131.0),2.0,2.6,fill=False,edgecolor=BK,linewidth=0.6))
    ax.text(45.4+k*2.1,132.3,g,ha="center",va="center",fontsize=5.4,color=BK)
box(44,53,123.5,128.5,INK,lw=0.9); ax.text(48.5,126.0,"DNA",ha="center",va="center",
        fontsize=6.4,color=INK,fontweight="bold")
th=np.linspace(0,4*np.pi,160)
ax.plot(48.5+1.8*np.sin(th),np.linspace(114,122,160),color=INK,linewidth=0.7)
ax.plot(48.5-1.8*np.sin(th),np.linspace(114,122,160),color=INK,linewidth=0.7)
box(44,53,108,113,ACC,lw=0.9); ax.text(48.5,110.5,"RNA",ha="center",va="center",
        fontsize=6.4,color=ACC,fontweight="bold")
th=np.linspace(0,3*np.pi,120)
ax.plot(48.5+1.6*np.sin(th),np.linspace(99,107,120),color=ACC,linewidth=0.7)
arrow(46,95,51,95,c=INK,lw=2.0,ms=11)

# ---------------- panel 2 ----------------
P2=(55,101); box(P2[0],P2[1],62,160,FRAME,lw=0.9)
numtag(P2[0]+4.5,156.5,2,ACC)
ax.text(79.5,157.4,"Evidence Transformation",ha="center",fontsize=7.0,color=ACC,fontweight="bold")
ax.text(78,154.4,"externally expressible artifact",ha="center",fontsize=6.2,color=BK,style="italic")
box(57,99,132,151,ACC,lw=0.8,ls=(0,(3,2)))
ax.text(78,148.6,"disclosure filtering",ha="center",fontsize=6.4,color=ACC,fontweight="bold")
FLT=[("signal\nabstraction",63.5),("temporal\nbinning",73.0),("identifier\nremoval",82.5),("feature\nencoding",92.0)]
for lab,x in FLT:
    ax.add_patch(Circle((x,142.0),3.1,fill=False,edgecolor=ACC,linewidth=0.8,zorder=3))
    ax.text(x,135.6,lab,ha="center",va="center",fontsize=5.5,color=BK,linespacing=1.15)
wave(x-2.2,x+2.2,142.0,0.8,ACC,seed=9)
ax.add_patch(Polygon([[63.5,143.6],[63.5,140.4]],closed=False,edgecolor=ACC,linewidth=0.7))
arrow(78,131.5,78,127.5,c=ACC,ls=(0,(2,2)),lw=0.8)
ax.text(80.0,129.8,"$R^{*}_{i,t}$",ha="left",va="center",fontsize=6.4,color=ACC)
d=[(78,123.0),(88,118.5),(78,114.0),(68,118.5)]
ax.add_patch(Polygon(d,closed=True,fill=False,edgecolor=INK,linewidth=0.9,zorder=3))
ax.text(78,118.5,"$A_R = 1$ ?",ha="center",va="center",fontsize=6.2,color=INK)
arrow(89,118.5,93.5,118.5,c=ACC,ls=(0,(2,2)),lw=0.8)
box(93,99.5,116.0,121.0,ACC,lw=0.8)
ax.text(96.2,118.5,"not\nreleased",ha="center",va="center",fontsize=5.6,color=ACC,linespacing=1.15)
arrow(78,113.0,78,109.0,c=GRN,lw=0.9)
ax.text(79.6,111.0,"yes",ha="left",va="center",fontsize=5.8,color=GRN)
box(57,99,74,108,ACC,lw=1.0)
ax.text(78,105.4,"signed evidence package  $\\tilde{R}_{i,t}$",ha="center",fontsize=6.6,
        color=ACC,fontweight="bold")
ax.text(78,102.6,"claim  ·  artifact  ·  explanation  ·  lineage",ha="center",fontsize=5.6,
        color=BK,style="italic")
ROWS=[("event time","12:04:31"),("phase","yellow \u2192 red"),("lane crossings","yes"),
      ("speed before","48 km/h"),("speed after","52 km/h")]
for k,(a,bv) in enumerate(ROWS):
    y=99.4-k*2.6
    ax.text(60.0,y,"\u2713",ha="left",va="center",fontsize=5.4,color=GRN)
    ax.text(62.5,y,a,ha="left",va="center",fontsize=5.6,color=BK)
    ax.text(96.5,y,bv,ha="right",va="center",fontsize=5.6,color=BK)
ax.text(60.0,85.0,"digital signature",ha="left",va="center",fontsize=6.0,color=ACC,fontweight="bold")
xx=np.linspace(80,96,180)
ax.plot(xx,85.0+1.3*np.sin((xx-80)*1.4)*np.exp(-((xx-88)/9)**2),color=BK,linewidth=0.7)
box(57,99,64,71,ACC,lw=0.9)
ax.text(78,67.5,"compact  ·  auditable  ·  verifiable",ha="center",va="center",
        fontsize=6.4,color=ACC,fontweight="bold")
arrow(101.5,111,106.5,111,c=INK,lw=2.0,ms=11)

# ---------------- panel 3 ----------------
P3=(108,143); box(P3[0],P3[1],62,160,FRAME,lw=0.9)
numtag(P3[0]+4.5,156.5,3,INK)
ax.text(127.0,157.4,"Distributed Review",ha="center",fontsize=7.0,color=INK,fontweight="bold")
ax.text(125.5,154.4,"and sharing",ha="center",fontsize=6.2,color=BK,style="italic")
def inst(cx,cy,lab,w=6.0,col=INK):
    h=w*0.62
    ax.add_patch(Polygon([[cx-w/2,cy+h*0.18],[cx,cy+h*0.52],[cx+w/2,cy+h*0.18]],closed=True,
                 fill=False,edgecolor=col,linewidth=0.8,zorder=4))
    ax.add_line(Line2D([cx-w/2,cx+w/2],[cy+h*0.18,cy+h*0.18],color=col,linewidth=0.8,zorder=4))
    for k in range(4):
        x=cx-w*0.34+k*(w*0.68/3)
        ax.add_line(Line2D([x,x],[cy-h*0.30,cy+h*0.12],color=col,linewidth=0.6,zorder=4))
    ax.add_line(Line2D([cx-w*0.5,cx+w*0.5],[cy-h*0.30,cy-h*0.30],color=col,linewidth=0.8,zorder=4))
    ax.text(cx,cy-4.6,lab,ha="center",va="center",fontsize=5.6,color=BK,linespacing=1.15)
inst(114.5,146.5,"traffic\nauthority"); inst(136.5,146.5,"department")
inst(113.0,123.5,"law and\nadjudication"); inst(138.0,123.5,"data\ncustodian")
ax.add_patch(FancyBboxPatch((117.5,128.0),16,10.0,boxstyle="round,pad=0,rounding_size=4.4",
             linewidth=0.9,edgecolor=INK,facecolor="white",zorder=3))
ax.text(125.5,135.0,"signed evidence",ha="center",fontsize=5.8,color=INK,fontweight="bold")
ax.text(125.5,132.2,"package",ha="center",fontsize=5.8,color=INK,fontweight="bold")
ax.add_patch(Rectangle((123.5,128.8),4.0,2.6,fill=False,edgecolor=ACC,linewidth=0.8,zorder=4))
for sx,sy,tx,ty,c in [(114.5,143.0,120.0,137.0,ACC),(136.5,143.0,131.5,137.0,ACC),
                      (113.0,127.1,119.0,129.5,INK),(138.0,127.1,132.5,129.5,INK)]:
    arrow(sx,sy,tx,ty,c=c,ls=(0,(2,2)),lw=0.7,ms=5)
ax.add_patch(FancyBboxPatch((116.0,106.2),19,8.4,boxstyle="round,pad=0,rounding_size=4.0",
             linewidth=0.9,edgecolor=INK,facecolor="white",zorder=3))
ax.text(125.5,111.4,"ledger and lineage",ha="center",fontsize=5.6,color=INK,fontweight="bold")
ax.text(125.5,108.6,"permissioned Merkle",ha="center",fontsize=5.2,color=BK,style="italic")
arrow(125.5,127.6,125.5,115.2,c=INK,ls=(0,(2,2)),lw=0.7,ms=5)
box(110,141,88,104,GRN,lw=0.8)
ax.text(125.5,101.2,"cross-node agreement",ha="center",fontsize=6.2,color=GRN,fontweight="bold")
ax.text(125.5,97.4,"$\\kappa \\in [0,1]$",ha="center",fontsize=7.0,color=BK)
ax.text(125.5,94.2,"agreement across institutional nodes",ha="center",fontsize=5.4,color=BK,style="italic")
NP=[(119,90.6,INK),(125.5,92.0,GRN),(132,90.6,ACC),(125.5,88.8,PUR)]
for x,y,c in NP: ax.add_patch(Circle((x,y),1.5,fill=False,edgecolor=c,linewidth=0.7,zorder=4))
for a in range(len(NP)):
    for bq in range(a+1,len(NP)):
        ax.add_line(Line2D([NP[a][0],NP[bq][0]],[NP[a][1],NP[bq][1]],color=GREY,linewidth=0.4,zorder=3))
box(110,141,64,86,GREY,lw=0.7)
LEG=[("attestation and review",INK,(0,(3,2))),("secure sharing",ACC,(0,(3,2))),("lineage record",BK,(0,(2,2)))]
for k,(lab,c,ls) in enumerate(LEG):
    y=81.5-k*4.4
    ax.add_line(Line2D([112.5,119.0],[y,y],color=c,linewidth=0.8,linestyle=ls))
    ax.text(120.5,y,lab,ha="left",va="center",fontsize=5.6,color=BK)
ax.text(125.5,67.0,"only admissibility calls are exchanged",ha="center",fontsize=5.6,
        color=BK,style="italic")
arrow(143.5,111,148.5,111,c=INK,lw=2.0,ms=11)

# ---------------- panel 4 ----------------
P4=(150,181.6); box(P4[0],P4[1],62,160,FRAME,lw=0.9)
numtag(P4[0]+3.6,157.6,4,PUR)
ax.text(168.9,158.0,"Protocol reliability",ha="center",fontsize=6.0,color=PUR,fontweight="bold")
ax.text(168.5,154.8,"four components",ha="center",fontsize=5.8,color=BK,style="italic")
COMP=[("triadic alignment",INK,r"$\Psi_{ER} \leftrightarrow \Psi_{DG} \leftrightarrow \Psi_{Case}$",
       "three views must agree",148.0),
      ("transition stability",ACC,r"$TS=\exp(-\Delta g^{\top} M \Delta g)$",
       "small change, small effect",125.0),
      ("admissibility survival",GRN,r"$SR=|\{E: E\in G(C)\}| \, / \, |\mathcal{E}|$",
       "candidates surviving the regime",102.0),
      ("governance coherence",PUR,r"$GC = 1 - v \, / \, n_{\mathrm{lin}}$",
       "retraceable and reviewable",79.0)]
for lab,col,eq,sub,ytop in COMP:
    box(152,180,ytop-19.0,ytop+3.0,col,lw=0.9)
    ax.text(166,ytop+0.6,lab,ha="center",fontsize=6.2,color=col,fontweight="bold")
    ax.text(166,ytop-5.6,eq,ha="center",fontsize=5.8,color=BK)
    ax.text(166,ytop-12.0,sub,ha="center",fontsize=5.2,color=BK,style="italic")
    if lab.startswith("admissibility"):
        for k in range(9):
            ax.plot([157+ (k%3)*1.9],[ytop-15.6+(k//3)*1.6],marker="o",markersize=1.5,color=GRN)
        ax.add_patch(Polygon([[173,ytop-14.0],[179,ytop-14.0],[176.6,ytop-17.4],[175.4,ytop-17.4]],
                     closed=True,fill=False,edgecolor=GRN,linewidth=0.7))

# ---------------- phase band ----------------
BAND=[(3,45,"SUPERVISORY PHASE","layers L1\u2013L2  ·  stages S1\u2013S6",
       r"$u \rightarrow E \rightarrow z=(g,q)$",
       r"$\Pi \cdot T_{\mathrm{gen}} \cdot T_{\mathrm{mut}} \cdot T_{\mathrm{rep}} \cdot A_g$",
       "genesis  ·  mutation  ·  repair",INK),
      (55,101,"EXTERNALIZATION BOUNDARY","layer L3  ·  stages S7\u2013S9",
       r"$E \rightarrow R^{*} \rightarrow R$",
       r"$\Phi \cdot A_R \cdot A_{\pi} \cdot A_{\mathrm{ctx}}$",
       "expression · immune and ectopic screening",ACC),
      (108,143,"EVIDENTIARY PHASE","layer L4  ·  stages S10\u2013S12",
       r"$R \rightarrow \tilde{R},\ \mathrm{Lin}(R)$",
       r"$\Lambda$: accept, certify, contest, revoke, regenerate",
       "agreement · apoptosis · memory · regeneration",INK),
      (150,181.6,"PROTOCOL RELIABILITY","measured across L1\u2013L4",
       r"$\mathrm{Rel}=A^{w_A} TS^{w_{TS}} SR^{w_{SR}} GC^{w_{GC}}$",
       "weakest-link aggregation",
       "no layer certifies alone",PUR)]
for x0,x1,title,sub,e1,e2,foot,col in BAND:
    box(x0,x1,36,59,col,lw=1.0)
    ax.text((x0+x1)/2,56.0,title,ha="center",fontsize=6.6,color=col,fontweight="bold")
    ax.text((x0+x1)/2,53.0,sub,ha="center",fontsize=5.5,color=BK)
    ax.text((x0+x1)/2,48.4,e1,ha="center",fontsize=6.6,color=BK)
    ax.text((x0+x1)/2,43.4,e2,ha="center",fontsize=5.0,color=col)
    ax.text((x0+x1)/2,38.6,foot,ha="center",fontsize=5.0,color=BK,style="italic")
    ax.add_line(Line2D([(x0+x1)/2,(x0+x1)/2],[36,33.5],color=GREY,linewidth=0.7))
ax.add_line(Line2D([24,165.8],[33.5,33.5],color=GREY,linewidth=0.8))
ax.add_line(Line2D([92,92],[33.5,30.5],color=GREY,linewidth=0.8))

ax.text(W/2,26.5,"Evidence certified, contested, revoked and regenerated",
        ha="center",fontsize=8.2,color=BK,fontweight="bold")
ax.text(W/2,22.4,"without raw data, without ground truth",
        ha="center",fontsize=8.2,color=PUR,fontweight="bold")
ax.text(W/2,18.4,"Raw traces stay inside the node; every institutional decision rests on an admissibility gate, "
        "an append-only lineage, and a protocol-level reliability score.",
        ha="center",fontsize=5.8,color=BK)
ax.text(W/2,15.2,"Ethical boundary: the declared constraint regime restricts what may be inferred, stored, "
        "or externalized at every layer.",ha="center",fontsize=5.8,color=BK,style="italic")

box(3,181.6,2.0,12.0,FRAME,lw=0.9)
L1=[("primary flow",INK,"-"),("attestation and review",INK,(0,(3,2))),
    ("secure information flow",ACC,(0,(3,2))),("lineage record",BK,(0,(2,2)))]
for k,(lab,c,ls) in enumerate(L1):
    y=9.6-k*2.2
    ax.add_line(Line2D([5.5,12.5],[y,y],color=c,linewidth=0.9,linestyle=ls))
    ax.text(13.5,y,lab,ha="left",va="center",fontsize=5.5,color=BK)
box(46,56,7.6,10.6,INK,lw=0.8); ax.text(51,9.1,"DNA",ha="center",va="center",fontsize=5.6,color=INK,fontweight="bold")
ax.text(57.5,9.1,"= internal evidentiary state",ha="left",va="center",fontsize=5.5,color=BK)
box(46,56,3.4,6.4,ACC,lw=0.8); ax.text(51,4.9,"RNA",ha="center",va="center",fontsize=5.6,color=ACC,fontweight="bold")
ax.text(57.5,4.9,"= externally expressible artifact",ha="left",va="center",fontsize=5.5,color=BK)
NOT=[(r"$u$","raw traces"),(r"$R^{*}$","candidate artifact"),(r"$R$","released artifact"),
     (r"$A_R$","admissibility gate")]
for k,(sy,tx) in enumerate(NOT):
    y=9.6-k*2.2
    ax.text(100,y,sy,ha="left",va="center",fontsize=5.8,color=BK)
    ax.text(105,y,": "+tx,ha="left",va="center",fontsize=5.5,color=BK)
NOT2=[(r"$\Delta g$","gene-coordinate displacement"),(r"$M$","quality weighting matrix"),
      (r"$\kappa$","agreement coefficient"),(r"$G(C)$","admissible region")]
for k,(sy,tx) in enumerate(NOT2):
    y=9.6-k*2.2
    ax.text(139,y,sy,ha="left",va="center",fontsize=5.8,color=BK)
    ax.text(150,y,": "+tx,ha="left",va="center",fontsize=5.2,color=BK)
fig.savefig("Figure_7_schematic.png",facecolor="white")
print("ok")
