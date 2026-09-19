"""Figure 14. Evaluation seeds versus generator draws.

Panel (a): deviation of the mean aggregate at 5, 10, 25, 50, 100, 200 and 400 evaluation
seeds from its four-hundred-seed value, with the 95% interval of the mean.
Panel (b): standard deviation of the aggregate across evaluation seeds at the same subset
sizes, on the per-event route (solid) and on the component route (dashed), against the
standard deviation across fifty generator draws on the same two routes (horizontal lines).

Reads seed_convergence.csv and seeds400.csv (run_seed_convergence.py) and
multiseed/rel_multiseed.csv, multiseed/rel_multiseed_summary.csv (rel_multiseed.py).
Writes figures/v52/Figure_14.png at 600 dpi.
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt

D = "../data/"; M = D + "multiseed/"
INK = "#1F3864"; ACC = "#C55A11"; BK = "#000000"; GR = "#7F7F7F"
MM = 1 / 25.4
OUT = "../figures/v52/Figure_14.png"
SUBSETS = [5, 10, 25, 50, 100, 200, 400]

cv = pd.read_csv(D + "seed_convergence.csv"); cv = cv[cv.metric == "rel"]
s4 = pd.read_csv(D + "seeds400.csv")
s4["comp"] = (s4.a_triad * s4.ts * s4.sr * s4.gc) ** 0.25
ms = pd.read_csv(M + "rel_multiseed_summary.csv").set_index("domain")
rm = pd.read_csv(M + "rel_multiseed.csv")
rm["comp"] = (rm.A_triad * rm.TS_mean * rm.SR_mean) ** 0.25

fig = plt.figure(figsize=(150 * MM, 58 * MM), dpi=600)
a1 = fig.add_axes([0.105, 0.25, 0.35, 0.62])
a2 = fig.add_axes([0.615, 0.25, 0.35, 0.62])

draw_sd = {}
for dom, col, mk in (("RLV", INK, "o"), ("Healthcare", ACC, "s")):
    c = cv[cv.domain == dom].set_index("n_seeds").loc[SUBSETS]
    ref = c.loc[400, "mean"]
    dev = c["mean"] - ref
    a1.fill_between(SUBSETS, dev - c["ci95"], dev + c["ci95"], color=col, alpha=0.15, linewidth=0)
    a1.plot(SUBSETS, dev, color=col, marker=mk, markersize=3.2, linewidth=1.2, label=dom)
    x = s4[s4.domain == dom].sort_values("seed")
    sd_pe = [x.rel.values[:n].std(ddof=1) for n in SUBSETS]
    sd_cp = [x.comp.values[:n].std(ddof=1) for n in SUBSETS]
    a2.plot(SUBSETS, sd_pe, color=col, marker=mk, markersize=3.2, linewidth=1.2)
    a2.plot(SUBSETS, sd_cp, color=col, linewidth=0.9, linestyle=(0, (3, 2)))
    draw_sd[dom] = (ms.loc[dom, "Rel_mean_sd"], rm[rm.domain == dom].comp.std(ddof=1))

pe = np.mean([v[0] for v in draw_sd.values()]); cp = np.mean([v[1] for v in draw_sd.values()])
a2.axhline(pe, color=GR, linewidth=1.0)
a2.axhline(cp, color=GR, linewidth=1.0, linestyle=(0, (3, 2)))
a2.text(5.3, pe + 0.0005, f"fifty generator draws, per-event route ({pe:.3f})", fontsize=5.8, color=BK)
a2.text(5.3, cp + 0.0005, f"fifty generator draws, component route ({cp:.3f})", fontsize=5.8, color=BK)
a2.text(5.3, 0.0046, "evaluation seeds, both routes (about 0.003)", fontsize=5.8, color=BK)

a1.axhline(0, color=GR, linewidth=0.8)
a1.set_ylim(-0.005, 0.005)
a1.set_ylabel("mean minus 400-seed mean", fontsize=6.8, color=BK)
a1.legend(fontsize=6.2, frameon=False, loc="upper right", labelcolor=BK)
a1.set_title("(a) the mean moves by at most 0.002", fontsize=7.0, color="#0F2545", fontweight="bold", pad=4, loc="left")
a2.set_ylim(0, 0.019)
a2.set_ylabel("standard deviation of the aggregate", fontsize=6.8, color=BK)
a2.set_title("(b) seeds vary it far less than data do", fontsize=7.0, color="#0F2545", fontweight="bold", pad=4, loc="left")
for a in (a1, a2):
    a.set_xscale("log"); a.set_xticks(SUBSETS); a.set_xticklabels([str(n) for n in SUBSETS])
    a.minorticks_off()
    a.set_xlabel("evaluation seeds", fontsize=6.8, color=BK)
    a.tick_params(labelsize=6.0, colors=BK)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.text(0.5, 0.015, "shaded: 95% interval of the mean; in (b), solid lines per-event route, dashed lines component route",
         ha="center", va="bottom", fontsize=5.6, color=BK, style="italic")
fig.savefig(OUT, facecolor="white", dpi=600)
plt.close(fig)
for dom, (p, c) in draw_sd.items():
    print(dom, "draw SD per-event %.4f component %.4f" % (p, c))
