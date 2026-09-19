"""Figure 15. Counts are raised until the estimates stop moving.

Three panels report the matched misrouting census of Section 3.8.1 across seed
runs; the fourth reports the seed convergence of the aggregate of Section 3.12.
Reads misrouting_saturation.csv, written by run_misrouting_saturation.py, and
seed_convergence.csv, written by run_seed_convergence.py. Both precede this
script in run_all.sh.

The figure is written at the size at which it is embedded in the manuscript,
4360 x 1195 pixels at 600 dpi, so that the script output, the file under
figures/ and the image in the paper are the same object. The crop is applied in
code rather than by hand; TRIM_TOP and TRIM_BOTTOM are the only constants that
carry it, and the assertion below fails rather than clipping content if a later
edit pushes a title or a caption into the trimmed band.
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

d = pd.read_csv("../data/misrouting_saturation.csv")
cv = pd.read_csv("../data/seed_convergence.csv")

INK = "#1F3864"; ACC = "#C55A11"; BK = "#000000"
MM = 1 / 25.4
TRIM_TOP, TRIM_BOTTOM = 55, 40          # to the embedded height of 1195 px
OUT = "Figure_15.png"

fig = plt.figure(figsize=(184.6 * MM, 57 * MM), dpi=600)
W, Y, H = 0.175, 0.315, 0.500
a1 = fig.add_axes([0.058, Y, W, H])
a2 = fig.add_axes([0.303, Y, W, H])
a3 = fig.add_axes([0.548, Y, W, H])
a4 = fig.add_axes([0.800, Y, W, H])

for dom, col, mk in [("RLV", INK, "o"), ("Healthcare", ACC, "s")]:
    s = d[d.domain == dom].sort_values("seed_runs")
    a1.plot(s.seed_runs, s.pairs, color=col, marker=mk, markersize=3.4, linewidth=1.3, label=dom)
    a2.plot(s.seed_runs, s.forfeit * 100, color=col, marker=mk, markersize=3.4, linewidth=1.3)
    a3.plot(s.seed_runs, s.worsened * 100, color=col, marker=mk, markersize=3.4, linewidth=1.3)
    c = cv[(cv.domain == dom) & (cv.metric == "rel")].sort_values("n_seeds")
    a4.plot(c.n_seeds, c["mean"], color=col, marker=mk, markersize=3.4, linewidth=1.3)
    a4.fill_between(c.n_seeds, c["mean"] - c.ci95, c["mean"] + c.ci95,
                    color=col, alpha=0.16, linewidth=0)

a1.axhline(32, color="#9A9A9A", linewidth=0.9, linestyle=(0, (4, 3)))
a1.text(5.5, 33.4, "32 unique cases", fontsize=6.0, color=BK)
a1.set_ylabel("matched pairs recovered", fontsize=7.0, color=BK); a1.set_ylim(0, 38)
a1.set_title("the cell is exhausted,\nnot sampled", fontsize=7.2, color="#0F2545", fontweight="bold", pad=4)
a1.legend(fontsize=6.4, frameon=False, loc="lower right", labelcolor=BK)

a2.set_ylabel("correction forfeited (%)", fontsize=7.0, color=BK); a2.set_ylim(0, 100)
a2.set_title("and the estimates\nsettle with it", fontsize=7.2, color="#0F2545", fontweight="bold", pad=4)

a3.set_ylabel("left worse than before repair (%)", fontsize=6.6, color=BK); a3.set_ylim(0, 40)
a3.set_title("as does\nthe harm", fontsize=7.2, color="#0F2545", fontweight="bold", pad=4)

a4.set_ylabel("aggregate reliability", fontsize=7.0, color=BK); a4.set_ylim(0.849, 0.862)
a4.set_yticks([0.850, 0.854, 0.858, 0.862])
a4.set_title("and the aggregate is\nalready settled at five", fontsize=7.2, color="#0F2545", fontweight="bold", pad=4)

for a in (a1, a2, a3):
    a.set_xscale("log"); a.set_xticks([5, 20, 80, 400]); a.set_xticklabels(["5", "20", "80", "400"])
    a.set_xlabel("seed runs", fontsize=7.0, color=BK)
    a.axvspan(80, 420, facecolor="#9A9A9A", alpha=0.10)

a4.set_xscale("log"); a4.set_xticks([5, 25, 100, 400]); a4.set_xticklabels(["5", "25", "100", "400"])
a4.set_xlabel("seeds", fontsize=7.0, color=BK)

for a in (a1, a2, a3, a4):
    a.tick_params(labelsize=6.4, colors=BK)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.spines["left"].set_color(BK); a.spines["bottom"].set_color(BK)

fig.text(0.5, 0.036,
         "beyond eighty seed runs the shaded region adds no new case and moves no reported quantity;\n"
         "the right panel shows the 95% interval of the aggregate over four hundred seeds",
         ha="center", va="bottom", fontsize=6.0, color=BK, style="italic", linespacing=1.3)

fig.savefig(OUT, facecolor="white")
from PIL import Image
plt.close(fig)

# Crop to the embedded size, and refuse to clip content rather than doing it silently.
im = Image.open(OUT)
w, h = im.size
ink = np.array(im.convert("L")) < 250
rows = np.where(ink.any(axis=1))[0]
assert rows[0] > TRIM_TOP and rows[-1] < h - TRIM_BOTTOM, (
    f"crop would clip content: ink spans rows {rows[0]}-{rows[-1]} of {h}, "
    f"trim is {TRIM_TOP}/{TRIM_BOTTOM}")
im.crop((0, TRIM_TOP, w, h - TRIM_BOTTOM)).save(OUT, dpi=(600, 600))
print("ok", Image.open(OUT).size)
