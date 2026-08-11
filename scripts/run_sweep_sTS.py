"""Transition stability across the declared stability scale.

The scale fixes the units in which a displacement is read rather than the
displacement itself, and Section 3.11 sweeps it as the fourth declared constant.
Writes sweep_sTS.csv.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep
B = _os.path.join(_HERE, "..", "data") + _os.sep
FIGDIR = _os.path.join(_HERE, "..", "figures") + _os.sep

import numpy as np, pandas as pd
G = ["g_S", "g_A", "g_D", "g_E"]
Q = ["q_S", "q_A", "q_D", "q_E"]
SCALES = [1, 2, 4, 6, 8, 12, 16, 24, 32]

sc = pd.read_csv(D + "scenarios.csv")
rows = []
for dom in ("RLV", "Healthcare"):
    d = sc[sc.domain == dom]
    piv = {t: d[d.stage_t == t].set_index(["event_id", "candidate_id"]) for t in (0, 1, 2)}
    idx = piv[0].index.intersection(piv[1].index).intersection(piv[2].index)
    g0 = piv[0].loc[idx, G].values.astype(float)
    g1 = piv[1].loc[idx, G].values.astype(float)
    g2 = piv[2].loc[idx, G].values.astype(float)
    q  = piv[0].loc[idx, Q].values.astype(float)
    wn = q / q.sum(axis=1, keepdims=True)
    for s in SCALES:
        w  = wn * s
        gm = np.exp(-((g1 - g0) * w * (g1 - g0)).sum(axis=1)).mean()
        mr = np.exp(-((g2 - g1) * w * (g2 - g1)).sum(axis=1)).mean()
        rows.append(dict(scale=s, domain=dom, TS_gm=float(gm), TS_mr=float(mr),
                         TS_mean=float((gm + mr) / 2), n=len(idx)))

t = pd.DataFrame(rows).sort_values(["scale", "domain"])
t.to_csv(D + "sweep_sTS.csv", index=False)
print(t.round(4).to_string(index=False))
