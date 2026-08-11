"""Saturation of the matched misrouting cell in the number of seed runs.

The cell is defined over unique combinations of event, candidate and mistaken
label, so raising the number of runs recovers cases rather than resampling them.
Figure 15 plots the result. Writes misrouting_saturation.csv.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
FIGDIR = _os.path.join(_HERE, "..", "figures") + _os.sep
import pandas as pd
import day6_ectopic_mislabel as M

import day6_ectopic_mislabel as M
GRID = [5, 10, 20, 40, 80, 150, 250, 400]

df  = pd.read_csv(D + "scenarios.csv")
pil = pd.read_csv(D + "pi_lookup.csv")

rows = []
for dom in ("RLV", "Healthcare"):
    for n in GRID:
        _, p = M.misrouting_cost(df, pil, dom, M.XI, M.ETA, list(range(n)))
        ach  = p.Xi_before - p.Xi_repair_true
        real = p.Xi_before - p.Xi_repair_declared
        rows.append(dict(domain=dom, seed_runs=n, pairs=len(p),
                         forfeit=float((ach.mean()-real.mean())/ach.mean()),
                         worsened=float((p.Xi_repair_declared > p.Xi_before).mean()),
                         cost_median=float(p.cost.median())))
        print(dom, n, len(p), flush=True)

pd.DataFrame(rows).to_csv(D + "misrouting_saturation.csv", index=False)
