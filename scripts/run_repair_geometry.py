"""Repair geometry and the stability ordering of Proposition P2'.

Section 3.9 asks how often the implemented corrective step is no longer than the
mutation that preceded it, and where the exceptions fall. Writes
repair_geometry.csv.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
D = _os.path.join(_HERE, "..", "data") + _os.sep

import numpy as np, pandas as pd
G = ["g_S", "g_A", "g_D", "g_E"]
Q = ["q_S", "q_A", "q_D", "q_E"]
S_TS = 8.0                      # declared stability scale, Table 1

sc = pd.read_csv(D + "scenarios.csv")
rows = []
for dom in ("RLV", "Healthcare"):
    d = sc[sc.domain == dom]
    piv = {t: d[d.stage_t == t].set_index(["event_id", "candidate_id"]) for t in (0, 1, 2)}
    idx = piv[0].index.intersection(piv[1].index).intersection(piv[2].index)
    dm, dr = [], []
    for k in idx:
        g0 = piv[0].loc[k, G].values.astype(float)
        g1 = piv[1].loc[k, G].values.astype(float)
        g2 = piv[2].loc[k, G].values.astype(float)
        q  = piv[0].loc[k, Q].values.astype(float)
        w  = q / q.sum() * S_TS          # weighting matrix of Equation (A4)
        dm.append(float((g1 - g0) @ (w * (g1 - g0))))
        dr.append(float((g2 - g1) @ (w * (g2 - g1))))
    dm, dr = np.array(dm), np.array(dr)
    hold = dr <= dm                       # condition of Proposition P2'
    rows.append(dict(domain=dom, n=len(idx), hold=float(hold.mean()),
                     ts_gm=float(np.exp(-dm).mean()), ts_mr=float(np.exp(-dr).mean()),
                     ts_gm_hold=float(np.exp(-dm[hold]).mean()),
                     ts_mr_hold=float(np.exp(-dr[hold]).mean()),
                     ts_gm_fail=float(np.exp(-dm[~hold]).mean()),
                     ts_mr_fail=float(np.exp(-dr[~hold]).mean()),
                     disp_m=float(dm.mean()), disp_r=float(dr.mean()),
                     disp_m_fail=float(dm[~hold].mean()),
                     disp_r_fail=float(dr[~hold].mean())))

t = pd.DataFrame(rows)
t.to_csv(D + "repair_geometry.csv", index=False)
print(t.round(4).to_string(index=False))
