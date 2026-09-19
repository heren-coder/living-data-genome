"""Figure 1 -- the Living Data Genome as a computational evidence ontology.
Redrawn in matplotlib (v1.3.7) from the Pages original so that panels, the central
helix and the lifecycle band are aligned and every label is at least 6 pt."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
MM=1/25.4; W,H=184.6,145.6
INK="#000000"; SUB="#222222"; ACC="#C55A11"; GREY="#8C8C8C"; LIGHT="#B8B8B8"
fig=plt.figure(figsize=(W*MM,H*MM),dpi=600)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor("white")
SERIF="DejaVu Serif"
def box(x0,x1,y0,y1,lw=0.9,r=1.6,ec=INK,z=2):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle=f"round,pad=0,rounding_size={r}",linewidth=lw,edgecolor=ec,facecolor="white",zorder=z))
def T(x,y,s,fs=6.6,w="normal",st="normal",c=INK,ha="center",va="center",ls=1.25,z=5):
    ax.text(x,y,s,fontsize=fs,fontweight=w,style=st,color=c,ha=ha,va=va,linespacing=ls,family=SERIF,zorder=z)
def arrow(x0,y0,x1,y1,c=INK,lw=0.9,ms=6):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",linewidth=lw,color=c,mutation_scale=ms,zorder=6,shrinkA=0,shrinkB=0))

# ---------------- helix drawing -----------------------------------------
def helix(path, amp, turns, lw=0.9, shade=True, rung_every=1.0, z=3):
    """Double helix along a parametric path: path(t)->(x,y), t in [0,1]."""
    t=np.linspace(0,1,1400); P=np.array([path(s) for s in t]); d=np.gradient(P,axis=0); n=np.stack([-d[:,1],d[:,0]],1); n/=np.linalg.norm(n,axis=1)[:,None]+1e-9
    ph=2*np.pi*turns*t; a=amp*np.sin(ph); b=-a
    A=P+n*a[:,None]; B=P+n*b[:,None]
    # depth shading: strand in front is darker
    for S,front in ((A,np.cos(ph)>0),(B,np.cos(ph)<=0)):
        col=np.where(front,0.15,0.55)
        for i in range(len(t)-1):
            ax.plot(S[i:i+2,0],S[i:i+2,1],color=(col[i],col[i],col[i]),lw=lw,solid_capstyle="round",zorder=z)
    # rungs
    for i in range(0,len(t),int(len(t)/(turns*8*rung_every))):
        ax.plot([A[i,0],B[i,0]],[A[i,1],B[i,1]],color=LIGHT,lw=0.45,zorder=z-1)

# ---------------- side panels -------------------------------------------
PT,PB=142.5,62.0
box(4,58,PB,PT,lw=1.0,r=2.0); box(126.6,180.6,PB,PT,lw=1.0,r=2.0)
T(31,136.6,"INFORMATION\nARCHITECTURE",fs=8.0,w="bold",ls=1.3); ax.add_line(Line2D([13,49],[129.6,129.6],color=INK,lw=0.6))
T(153.6,136.6,"CONTEXTUAL\nREGULATION",fs=8.0,w="bold",ls=1.3); ax.add_line(Line2D([136,171],[129.6,129.6],color=INK,lw=0.6))
def card(x0,x1,y0,y1,title,sub):
    box(x0,x1,y0,y1,lw=0.8,r=1.2,z=3); cy=(y0+y1)/2
    T((x0+x1)/2,cy+4.0,title,fs=7.2,w="bold",ls=1.2); T((x0+x1)/2,cy-4.2,sub,fs=6.2,st="italic",c=SUB,ls=1.3)
card(9,53,101,125,"DNA-level\nInternal Evidence","Structured, encoded,\nand semantically typed.")
arrow(31,101,31,93.5,lw=0.8)
card(9,53,69,93,"Governed\nRNA Expression","Selective release under\ngovernance policy.")
card(131.6,175.6,106,125,"Context-conditioned\nGene Dominance","Prioritizes relevant evidence\nunder the current context.")
arrow(153.6,106,153.6,102,lw=0.8)
card(131.6,175.6,86,101.5,"Ectopic Expression","Flags profiles discordant\nwith the declared context.")
arrow(153.6,86,153.6,82,lw=0.8)
card(131.6,175.6,66,81.5,"Repair Routing","Routes corrective actions\nunder governing policies.")
T(31,55.5,"Defines the internal representation\nof evidence and its governed\nrelease across the boundary.",fs=6.4,st="italic",c=SUB,ls=1.35)
T(153.6,55.5,"Adapts evidence behaviour to context,\nflags context mismatch, and routes\nrepair under governing policies.",fs=6.4,st="italic",c=SUB,ls=1.35)

# ---------------- central organism --------------------------------------
CX,CY=92.3,102.0; RX,RY=32.5,39.0
def loop(t):  # egg-shaped closed path, apex at the bottom
    th=2*np.pi*t+np.pi/2
    y=CY+RY*np.sin(th); x=CX+RX*np.cos(th)*(0.74+0.26*(np.sin(th)+1)/2)
    return (x,y)
helix(loop,2.2,12,lw=0.85,rung_every=0.7)
T(CX,CY+13.0,"LIVING\nDATA\nGENOME",fs=12.5,w="bold",ls=1.15)
ax.add_line(Line2D([CX-16,CX+16],[CY-1.0,CY-1.0],color=INK,lw=0.6))
T(CX,CY-5.5,"Evidence Organism",fs=8.0,st="italic")
T(CX,CY-13.5,"A governed computational entity\noperating under limited\nobservability.",fs=6.6,st="italic",c=SUB,ls=1.35)
# connectors to the side panels, symmetric
for x0,x1 in ((58,CX-RX*0.80),(126.6,CX+RX*0.80)):
    helix(lambda t,x0=x0,x1=x1:(x0+(x1-x0)*t,CY), 1.2, 2, lw=0.7, z=2)
    ax.add_patch(Circle((x0,CY),1.1,facecolor="white",edgecolor=INK,lw=0.7,zorder=4))
# stem to the lifecycle band
ax.add_line(Line2D([CX,CX],[CY-RY-0.5,50.5],color=INK,lw=0.9,zorder=3))
ax.add_patch(Circle((CX,50.5),1.1,facecolor="white",edgecolor=INK,lw=0.7,zorder=4))

# ---------------- lifecycle band ----------------------------------------
BT,BB=48.0,3.0
box(3,181.6,BB,BT,lw=1.0,r=2.0)
T(92.3,43.8,"GOVERNED EVIDENCE LIFECYCLE",fs=8.4,w="bold")
stages=[("Genesis","(Seeding)","Bounded seeding\nfrom traces."),("Mutation","(Variation)","Bounded change\nin the regime."),
        ("Repair","(Correction)","Admissibility\nis restored."),("Expression","(Release)","Release at the\nboundary."),
        ("Apoptosis","(Revocation)","Validity is\nwithdrawn."),("Regeneration","(Re-issue)","A corrected\nre-issue."),
        ("Memory","(Lineage)","Append-only,\nnot rewritten.")]
n=7; gap=2.8; w=(181.6-3-2*4.5-(n-1)*gap)/n; y0,y1=19.5,38.5
xs=[7.5+i*(w+gap) for i in range(n)]
for i,(x,(a,b,c)) in enumerate(zip(xs,stages)):
    box(x,x+w,y0,y1,lw=0.8,r=1.2,z=3); cx=x+w/2
    T(cx,35.0,a,fs=7.0,w="bold"); T(cx,31.6,b,fs=6.2); ax.add_line(Line2D([x+3,x+w-3],[29.3,29.3],color=INK,lw=0.4,zorder=4))
    T(cx,24.6,c,fs=6.2,ls=1.3)
    if i<n-1: arrow(x+w+0.3,29.0,x+w+gap-0.3,29.0,lw=0.8,ms=5)
# boundary between expression and apoptosis
bx=xs[4]-gap/2
ax.add_line(Line2D([bx,bx],[10.0,BT-6.5],color=ACC,lw=0.9,linestyle=(0,(4,2.5)),zorder=2))
T(bx-1.2,40.6,"local",fs=6.2,st="italic",c=ACC,ha="right"); T(bx+1.2,40.6,"institution-facing",fs=6.2,st="italic",c=ACC,ha="left")
# strand: helix on the local side, relaxed line on the institution side
helix(lambda t:(7.5+(bx-2.0-7.5)*t,13.0),2.2,4,lw=0.7,z=2)
tt=np.linspace(bx+1.0,181.6-7.5,300); ax.plot(tt,13.0+0.9*np.sin(np.linspace(0,5*np.pi,300))*np.linspace(1,0.3,300),color=LIGHT,lw=0.9,zorder=2)
T(92.3,7.0,"From bounded genesis to an append-only lineage, with integrity-preserving, individually certified transitions.",fs=6.4,st="italic",c=SUB)
fig.savefig("../figures/Figure_1.png",facecolor="white"); print("ok")
