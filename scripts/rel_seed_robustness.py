"""Table 3.2: seed variability of the aggregate Rel and of the supervised baseline.
The pipeline is deterministic given a seed; the only between-seed variation is the
case-facing noise draw of Section 3.1 and the baseline split, so Rel varies through
triadic alignment alone. A_triad samples are taken from the same helper used for the
ablations, so a single seed convention governs every five-seed figure in Section 3.
Outputs: data/rel_seed_robustness.csv
"""
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import run_baseline
from ablation_significance import full_model_samples, SEED_STRIDE, N_SEEDS

D = "../data/"
SEEDS = [RNG_SEED + i * SEED_STRIDE for i in range(N_SEEDS)]
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}

r = pd.read_csv(D + "rel_summary.csv").set_index("domain")
df = pd.read_csv(D + "scenarios.csv")
rows = []
for dom in ["RLV", "Healthcare"]:
    sub = df[df.domain == dom]
    ts, sr, gc = float(r.loc[dom, "TS_mean"]), float(r.loc[dom, "SR_mean"]), 1.0
    fm = full_model_samples(sub, CFG[dom], SEEDS)
    at = np.array(fm["A_triad"]); ba = np.array(fm["baseline"])
    f1 = np.array([run_baseline(sub, seed=s)["f1"] for s in SEEDS])
    rel = (at * ts * sr * gc) ** 0.25
    rows.append(dict(domain=dom, n_seeds=N_SEEDS,
                     A_triad_mean=at.mean(), A_triad_std=at.std(ddof=1),
                     baseline_balanced_accuracy_mean=ba.mean(),
                     baseline_balanced_accuracy_std=ba.std(ddof=1),
                     baseline_f1_mean=f1.mean(), baseline_f1_std=f1.std(ddof=1),
                     Rel_mean=rel.mean(), Rel_std=rel.std(ddof=1),
                     Rel_min=rel.min(), Rel_max=rel.max()))
out = pd.DataFrame(rows)
out.to_csv(D + "rel_seed_robustness.csv", index=False)
print(out.round(4).to_string(index=False))
print("Saved: data/rel_seed_robustness.csv")
