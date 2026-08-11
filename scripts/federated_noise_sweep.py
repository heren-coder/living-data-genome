"""
federated_noise_sweep.py -- Figure 3.7, Fleiss' kappa across institutional
calibration variance.

Why this script exists
----------------------
`day5_federated_crypto.py` evaluates federated agreement at a single node noise
level (node_noise_std = 0.015), so `federated_agreement.csv` is one point and the
sweep behind Figure 3.7 could not be reproduced from the workspace.

The manuscript expresses calibration noise relative to each domain's operating
tolerance (0.33 for RLV, 0.31 for the healthcare vignette), so the reference
point of Table 3.5 is 0.015 / 0.33 = 4.5 per cent. This script sweeps the
relative variance and reports the five-seed mean and standard deviation, which
is what the shaded band of Figure 3.7 shows.

The substantive reading the figure supports: agreement is high and stable in the
nominal operating range (3 to 9 per cent), and beyond roughly 20 per cent it not
only declines but becomes strongly seed-dependent, the standard deviation
widening from about 0.04 to about 0.29.

Inputs : ../scenarios.csv
Outputs: data/sweep_node_noise.csv
"""

import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from day5_federated_crypto import federated_agreement_demo

SEEDS = [RNG_SEED + i * 100 for i in range(5)]
RELATIVE_GRID = [0, 3, 5, 9, 10, 15, 20, 25, 30, 40]   # per cent of operating tolerance
TOLERANCE = {"RLV": 0.33, "Healthcare": 0.31}          # Table 3.8, feasibility tolerance
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]
REFERENCE_PCT = 4.5                                     # 0.015 / 0.33, Table 3.5


def main():
    df = pd.read_csv("../data/scenarios.csv")

    rows = []
    print("=== Figure 3.7, Fleiss' kappa across calibration variance ===")
    print(f"{'rel %':>6} | {'RLV kappa':>16} | {'Healthcare kappa':>18}")
    for pct in RELATIVE_GRID:
        line = {}
        for domain, config in CONFIG:
            sub = df[df.domain == domain]
            std = pct / 100.0 * TOLERANCE[domain]
            k, u, m = [], [], []
            for s in SEEDS:
                r = federated_agreement_demo(domain, sub, config, n_nodes=4,
                                             node_noise_std=std, seed=s)
                k.append(r["fleiss_kappa"])
                u.append(r["unanimous_rate"])
                m.append(r["mean_admissible_rate"])
            k, u, m = np.array(k), np.array(u), np.array(m)
            line[domain] = (k.mean(), k.std(ddof=1))
            rows.append({
                "domain": domain, "relative_variance_pct": pct, "node_noise_std": std,
                "fleiss_kappa_mean": k.mean(), "fleiss_kappa_sd": k.std(ddof=1),
                "unanimous_rate_mean": u.mean(), "unanimous_rate_sd": u.std(ddof=1),
                "mean_admissible_rate": m.mean(),
                "regime": ("nominal" if 3 <= pct <= 9 else
                           "stress_test" if pct >= 20 else "intermediate"),
            })
        print(f"{pct:>6} | {line['RLV'][0]:>8.3f} +/- {line['RLV'][1]:.3f} | "
              f"{line['Healthcare'][0]:>10.3f} +/- {line['Healthcare'][1]:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("../data/sweep_node_noise.csv", index=False)
    print(f"\nReference point of Table 3.5 is {REFERENCE_PCT} per cent relative variance.")
    print("Saved: data/sweep_node_noise.csv")


if __name__ == "__main__":
    main()
