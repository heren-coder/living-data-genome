"""Coherence-flag sweep over the two declared corridors.

Equation (2.26) flags a candidate as coherent when the level component clears
one threshold and the balance component clears another. Section 3.2 reports how
far both may be raised before either configuration stops clearing them. Writes
sweep_coh_thresholds.csv.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG
import rel_computation as R

from generator import RLV_CONFIG, HEALTHCARE_CONFIG
import rel_computation as R
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
LEV = np.round(np.arange(0.50, 0.96, 0.01), 2)
BAL = np.round(np.arange(0.50, 0.96, 0.01), 2)

sc = pd.read_csv(D + "scenarios.csv")
tr = pd.read_csv(D + "raw_traces.csv")

measured = {}
for dom, cfg in CFG.items():
    cl = R.compute_cross_layer(sc[sc.domain == dom], tr[tr.domain == dom], cfg)
    measured[dom] = (float(cl["CL_lev"]), float(cl["CL_bal"]))
    print(f"{dom}: level {measured[dom][0]:.3f}  balance {measured[dom][1]:.3f}")

rows = []
for tl in LEV:
    for tb in BAL:
        rec = dict(tau_lev=float(tl), tau_bal=float(tb))
        for dom in CFG:
            lev, bal = measured[dom]
            rec[f"pass_{dom}"] = int(lev >= tl and bal >= tb)
        rec["pass_both"] = int(rec["pass_RLV"] and rec["pass_Healthcare"])
        rows.append(rec)

t = pd.DataFrame(rows)
t.to_csv(D + "sweep_coh_thresholds.csv", index=False)
ok = t[t.pass_both == 1]
print(f"points {len(t)} | both configurations clear up to "
      f"tau_lev {ok.tau_lev.max():.2f}, tau_bal {ok.tau_bal.max():.2f}")
