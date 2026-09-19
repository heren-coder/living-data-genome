"""
fix_graphical_abstract_labels.py -- healthcare coordinate names in the
graphical abstract, spelled as Table 4 spells them.

Why this script exists
----------------------
The graphical abstract is hand-drawn and has no source file, so a label change
would otherwise be an untracked manual edit. It carried the healthcare
coordinates abbreviated -- AccessEscalation, RecordDensity, AccessControl --
while Table 4 and Supplementary Table S3.4 name them AccessEscalationRate,
ConcurrentRecordDensity and AccessControlRegime. A reader comparing the poster
with the table met three different names for the same three coordinates.

The full names do not fit at the original size: the columns are 437 px apart
and ConcurrentRecordDensity needs 474 px. The row is therefore redrawn one step
smaller, at 37 px against roughly 41 px before, in PT Sans, which matches the
letterforms of the hand-drawn original. Only the one row of orange labels is
touched; every other pixel is carried through unchanged.

Input : figures/graphical_abstract_v52_original.png  (labels as first drawn)
Output: figures/graphical_abstract_v52.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures") + os.sep
SRC = FIG + "graphical_abstract_v52_original.png"
DST = FIG + "graphical_abstract_v52.png"

BG = (238, 242, 248)          # light-blue panel fill
COL = (197, 90, 17)           # the accent orange used throughout the figure
ROW = (147, 1006, 1884, 1053) # the label row, inside the panel border
CENTRES = [364, 802, 1241, 1675]
LABELS = ["AccessEscalationRate", "AuditCoverage",
          "ConcurrentRecordDensity", "AccessControlRegime"]
FONT = "/System/Library/Fonts/Supplemental/PTSans.ttc"
SIZE, BASELINE, RIGHT_EDGE = 37, 1029, 1878

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
d.rectangle(list(ROW), fill=BG)
f = ImageFont.truetype(FONT, SIZE)
for c, t in zip(CENTRES, LABELS):
    w = d.textlength(t, font=f); bb = f.getbbox(t)
    x = min(c - w / 2, RIGHT_EDGE - w)
    d.text((x, BASELINE - (bb[1] + bb[3]) / 2), t, font=f, fill=COL)
im.save(DST)
print("written:", DST)
