"""
sigma_sweep.py -- sensitivity of triadic alignment and the cross-layer coherence
proxy to the case-facing noise constant sigma.

Why this script exists
----------------------
Both `compute_a_triad` (Eq. A2) and `compute_cross_layer` (Eq. 2.25) construct
the case-facing view as

    context_mean[c] + Normal(0, sigma),   sigma = 0.16

sigma is declared in Table 3.8 but is not calibrated to anything, and Section 3.8
sweeps three protocol constants (delta, w, eps_P) without sweeping this one. Two
consequences follow, and both are measured here.

  1. A_triad is a smooth monotone function of sigma over roughly [0.32, 0.81].
     The reported 0.714 is therefore a value the protocol sets rather than one it
     measures. The genesis-repair pairwise CKA is invariant under sigma, since
     that pair never touches the case view, which is the concrete instance of the
     warning already stated in Section 2.5.6: when two of three views share an
     underlying construction, their pairwise similarity reports the construction.

  2. The level agreement between A_triad and the coherence proxy holds at every
     sigma, but the balance contrast that Section 3.1 calls "the substantive
     result of this subsection" REVERSES SIGN below sigma ~ 0.08. At sigma = 0 the
     coherence balance is lower than the triadic balance; at the declared 0.16 it
     is higher. The claim is therefore conditional on sigma, and the condition is
     currently undeclared.

Inputs : ../scenarios.csv , raw_traces.csv (for the layer stack)
Outputs: data/sigma_sweep.csv
"""

import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import compute_a_triad, compute_cross_layer

GRID = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.32, 0.50, 1.00]
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]


def main():
    df = pd.read_csv("../data/scenarios.csv")
    traces = pd.read_csv("../data/raw_traces.csv")

    rows = []
    for domain, config in CONFIG:
        sub = df[df.domain == domain]
        tr = traces[traces.domain == domain]
        print(f"\n=== {domain} ===")
        print(f"{'sigma':>6} | {'A_tri lev':>9} {'A_tri bal':>9} | "
              f"{'CL lev':>7} {'CL bal':>7} | {'bal diff':>8} | {'CKA g-r':>8}")
        for s in GRID:
            a = compute_a_triad(sub, config, case_facing_noise_std=s, seed=RNG_SEED + 7)
            v = np.array([a["CKA_genesis_repair"],
                          a["CKA_genesis_context"],
                          a["CKA_repair_context"]])
            at_lev = float(np.prod(v) ** (1 / 3))   # same aggregation as CL_lev
            at_bal = float(v.min() / v.max())
            c = compute_cross_layer(sub, tr, config, case_facing_noise_std=s,
                                    seed=RNG_SEED + 7)
            print(f"{s:>6.2f} | {at_lev:>9.4f} {at_bal:>9.4f} | "
                  f"{c['CL_lev']:>7.4f} {c['CL_bal']:>7.4f} | "
                  f"{c['CL_bal'] - at_bal:>+8.4f} | {v[0]:>8.4f}")
            rows.append({
                "domain": domain, "sigma": s,
                "A_triad_mean": a["A_triad"],
                "A_triad_level": at_lev, "A_triad_balance": at_bal,
                "CKA_genesis_repair": v[0],
                "CKA_genesis_case": v[1], "CKA_repair_case": v[2],
                "CL_level": c["CL_lev"], "CL_balance": c["CL_bal"],
                "balance_gap_CL_minus_Atriad": c["CL_bal"] - at_bal,
                "operating_point": s == 0.16,
            })

    out = pd.DataFrame(rows)
    out.to_csv("../data/sigma_sweep.csv", index=False)

    print("\nSign of the balance contrast (CL balance minus A_triad balance):")
    for domain, _ in CONFIG:
        d = out[out.domain == domain].sort_values("sigma")
        flip = d[d.balance_gap_CL_minus_Atriad > 0].sigma.min()
        print(f"  {domain:<11} negative below sigma ~ {flip:.2f}, positive at and above it")
    print("\nSaved: data/sigma_sweep.csv")


if __name__ == "__main__":
    main()
