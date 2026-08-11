"""Matched misrouting counterfactual over four hundred seed runs.

Section 3.8.1 reports the cost of repairing a corrupted candidate under its
declared label rather than its true one. The forty-run figure of the earlier
pipeline is extended here so that the matched cell can be shown to be exhausted
rather than sampled; see run_misrouting_saturation.py for the sweep itself.
Writes misrouting_400.csv and misrouting_pairs_400.csv.
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
N_RUNS = 400

df  = pd.read_csv(D + "scenarios.csv")
pil = pd.read_csv(D + "pi_lookup.csv")

summaries, pairs = [], []
for dom in ("RLV", "Healthcare"):
    s, p = M.misrouting_cost(df, pil, dom, M.XI, M.ETA, list(range(N_RUNS)))
    summaries.append(s); pairs.append(p)

pd.DataFrame(summaries).to_csv(D + "misrouting_400.csv", index=False)
P = pd.concat(pairs); P.to_csv(D + "misrouting_pairs_400.csv", index=False)

for dom in ("RLV", "Healthcare"):
    p = P[P.domain == dom]
    ach  = p.Xi_before - p.Xi_repair_true
    real = p.Xi_before - p.Xi_repair_declared
    print(f"{dom}: pairs {len(p)} | forfeited "
          f"{(ach.mean()-real.mean())/ach.mean():.3f} | worse than before "
          f"{(p.Xi_repair_declared > p.Xi_before).mean():.3f} | "
          f"median cost {p.cost.median():+.4f}")
