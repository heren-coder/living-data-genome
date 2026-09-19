"""Seed convergence of the aggregate and its components.

Section 3.12 states that most statistics are computed over five seeds because the
pipeline is deterministic and the only between-seed variation is the case-facing
draw. This script shows that rather than asserting it: the reliability contract is
recomputed under four hundred seeds and the running mean and interval are recorded
at increasing subset sizes. Writes seeds400.csv and seed_convergence.csv.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
import rel_computation as R

D = "../data/"
N_SEEDS = 400
SUBSETS = [5, 10, 25, 50, 100, 200, 400]
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}

sc = pd.read_csv(D + "scenarios.csv")
rows = []
for dom, cfg in CFG.items():
    d = sc[sc.domain == dom]
    for s in range(N_SEEDS):
        summ, _, _ = R.compute_rel(dom, d, cfg, seed=RNG_SEED + s)
        rows.append(dict(domain=dom, seed=s,
                         rel=float(summ["Rel_mean"]), a_triad=float(summ["A_triad"]),
                         ts=float(summ["TS_mean"]), sr=float(summ["SR_mean"]),
                         gc=1.0))
        if (s + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv("../data/seeds400.csv", index=False)
            print(f"  {dom} {s+1}/{N_SEEDS}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("../data/seeds400.csv", index=False)

conv = []
for dom in CFG:
    x = df[df.domain == dom]
    for n in SUBSETS:
        for col in ("rel", "a_triad", "ts", "sr"):
            v = x[col].values[:n]
            conv.append(dict(domain=dom, n_seeds=n, metric=col,
                             mean=float(v.mean()), sd=float(v.std(ddof=1)),
                             ci95=float(1.96 * v.std(ddof=1) / np.sqrt(n)),
                             vmin=float(v.min()), vmax=float(v.max())))
c = pd.DataFrame(conv)
c.to_csv("../data/seed_convergence.csv", index=False)

print(c[c.metric == "rel"].round(4).to_string(index=False))
