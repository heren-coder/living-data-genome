"""
make_v52_figures.py -- assemble the figure files of the revised manuscript (v52)
from the figure scripts' outputs, under the v52 numbering.

The revision changed figure numbering: Figures 11+12 and 13+14 of the submitted
version were combined into two-panel figures, Figure 15 moved to Supplementary
Figure S2.1, Figures 17-18 moved to Supplementary Figures S7.1-S7.2, and Figure 16
became Figure 13. Panels are unchanged; this script only stacks and renames.

Reads  figures/Figure_NN.png (written by fig*.py, v51 numbering)
Writes figures/v52/Figure_NN.png and figures/v52/Figure_SN.png
"""
import os, shutil
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); OUT = os.path.join(FIG, "v52"); os.makedirs(OUT, exist_ok=True)
def stack(a, b, out, gap=60):
    A = Image.open(os.path.join(FIG, f"Figure_{a}.png")).convert("RGB"); B = Image.open(os.path.join(FIG, f"Figure_{b}.png")).convert("RGB")
    w = max(A.width, B.width); C = Image.new("RGB", (w, A.height + B.height + gap), "white"); C.paste(A, (0, 0)); C.paste(B, (0, A.height + gap))
    C.save(os.path.join(OUT, out), dpi=(600, 600))
MAP = {  # v52 name : v51 source
    "Figure_1.png": "Figure_1.png", "Figure_2.png": "Figure_2.png", "Figure_3.png": "Figure_3.png", "Figure_4.png": "Figure_4.png",
    "Figure_5.png": "Figure_5.png", "Figure_6.png": "Figure_6c.png", "Figure_7.png": "Figure_7_original_600dpi.png", "Figure_8.png": "Figure_8.png",
    "Figure_9.png": "Figure_9.png", "Figure_10.png": "Figure_10.png", "Figure_13.png": "Figure_16.png",
    "Figure_S2_1.png": "Figure_15.png", "graphical_abstract.png": "graphical_abstract_v52.png", "Figure_S7_1.png": "Figure_17.png", "Figure_S7_2.png": "Figure_18.png",
}
for dst, src in MAP.items():
    p = os.path.join(FIG, src)
    if os.path.exists(p): shutil.copyfile(p, os.path.join(OUT, dst))
    else: print("missing source:", src)
stack(11, 12, "Figure_11.png")   # (a) ablation impact, (b) governance operating map
stack(13, 14, "Figure_12.png")   # (a) federated agreement, (b) context-mismatch screen
print("v52 figures written to", OUT)
