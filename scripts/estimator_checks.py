"""Estimator notes for Tables 3 and 4 and for the stability term of Equation (2.11).

1. Triadic alignment. Table 3 reports the single pass; the Table 4 headings report the mean over the five
   evaluation seeds 42 + 100i. The first of those seeds reproduces the single pass, and the difference is
   of the order of the standard deviation across seeds.
2. Transition stability. The single stability term is the arithmetic mean of the two governed transitions,
   per candidate, then over the candidates of an event and over events. The geometric mean of the two
   transitions is reported next to it, to show that the choice leaves the reported stability, and the
   sensitivity argument of Section 3.10, unchanged.

Reads  data/scenarios.csv
Writes data/estimator_checks.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
import rel_computation as R  # noqa: E402
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED  # noqa: E402

D = os.path.join(HERE, "..", "data") + os.sep
SEEDS = [RNG_SEED + 100 * i for i in range(5)]   # as rel_seed_robustness.py
df = pd.read_csv(D + "scenarios.csv")
rows = []
for dom, cfg in (("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)):
    d = df[df.domain == dom]
    single = R.compute_a_triad(d, cfg)["A_triad"]                       # default seed, as Table 3
    seeds = [R.compute_a_triad(d, cfg, seed=s + 7)["A_triad"] for s in SEEDS]
    ts = R.compute_ts_per_event(d)
    ts["TS_geometric"] = np.sqrt(ts.TS_genesis_to_mutation * ts.TS_mutation_to_repair)
    ev = ts.groupby("event_id")[["TS_event", "TS_geometric"]].mean()
    rows.append(dict(domain=dom, A_triad_single_pass=single, A_triad_first_seed=seeds[0],
                     A_triad_five_seed_mean=float(np.mean(seeds)), A_triad_five_seed_sd=float(np.std(seeds, ddof=1)),
                     TS_arithmetic=float(ev.TS_event.mean()), TS_geometric=float(ev.TS_geometric.mean())))
out = pd.DataFrame(rows)
out.to_csv(D + "estimator_checks.csv", index=False)
print(out.round(4).to_string(index=False))
