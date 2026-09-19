"""
governance_multiseed.py -- Table 3.4 over five seeds, the random-redraw control,
and the tightening sweep behind Figure 3.6.

Why this script exists
----------------------
`day5_governance.py` is single-seed and computes no random-redraw rate, so
neither the "mean +/- SD over 5 seeds" of Table 3.4 nor the redraw control could
be reproduced from the workspace. It also selects the representative
generation-1 artifact deterministically:

    repair[repair.A_g].sort_values("candidate_id").groupby("event_id").first()

while the caption of Table 3.4 states "Representative artifact drawn at random
per event". Under the deterministic selection the revoke rate has zero variance,
which is inconsistent with the published 0.307 +/- 0.039. Under random selection
every entry of Table 3.4 reproduces within one standard deviation. The published
table is right and the workspace script is the older one.

Everything else -- the tightened regime C+, the mutation-repair regeneration
step, the Cons(L) scoring of GC -- is taken unchanged from day5_governance.py.

Inputs : ../scenarios.csv , data/rel_summary.csv
Outputs: data/governance_multiseed.csv , data/sweep_tightening.csv
"""

import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from day5_governance import tightened_bounds, is_admissible, rel_with

GENES = ["S", "A", "D", "E"]
SEEDS = [RNG_SEED + i * 100 for i in range(5)]
TIGHTENING = 0.05
TIGHTEN_GRID = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]


def one_arm(sub, config, tightening, seed):
    """One governance pass. Representative artifact drawn at random per event."""
    rng = np.random.default_rng(seed + 999)
    repair = sub[sub.stage_t == 2]
    admissible = repair[repair.A_g]
    gen1 = admissible.groupby("event_id").sample(n=1, random_state=seed).reset_index(drop=True)

    n = len(gen1)
    revoked, kept = [], 0
    for _, r in gen1.iterrows():
        c_plus = tightened_bounds(config, r.context_label, tightening=tightening)
        g = np.array([r.g_S, r.g_A, r.g_D, r.g_E])
        if is_admissible(g, c_plus):
            kept += 1
        else:
            revoked.append((g, c_plus))

    n_rev = len(revoked)
    regen_ok = redraw_ok = 0
    for g0, c_plus in revoked:
        mutated = np.clip(g0 + rng.normal(0, 0.14, 4), 0.0, 1.0)
        centre = np.array([(c_plus[k][0] + c_plus[k][1]) / 2 for k in GENES])
        repaired = np.clip(mutated + 0.55 * (centre - mutated) + rng.normal(0, 0.042, 4), 0.0, 1.0)
        if is_admissible(repaired, c_plus):
            regen_ok += 1
        # unconditioned uniform redraw: the chance control of Table 3.4
        if is_admissible(rng.random(4), c_plus):
            redraw_ok += 1

    return {
        "n_gen1": n,
        "revoke_rate": n_rev / n,
        "regeneration_success_rate": regen_ok / n_rev if n_rev else np.nan,
        "random_redraw_success_rate": redraw_ok / n_rev if n_rev else np.nan,
        "SR_discard": kept / n,
        "GC_discard": 1 - n_rev / n,
        "SR_regenerate": (kept + regen_ok) / n,
    }


def main():
    df = pd.read_csv("../data/scenarios.csv")
    rel = pd.read_csv("../data/rel_summary.csv").set_index("domain")

    # ---------------- Table 3.4, five seeds ----------------
    rows = []
    print("=== Table 3.4, five seeds, representative artifact drawn at random ===")
    for domain, config in CONFIG:
        sub = df[df.domain == domain]
        a, ts = rel.loc[domain, "A_triad"], rel.loc[domain, "TS_mean"]
        arms = [one_arm(sub, config, TIGHTENING, s) for s in SEEDS]
        relA = np.array([rel_with(a, ts, 1.0, 1.0) for _ in arms])
        relB = np.array([rel_with(a, ts, x["SR_discard"], x["GC_discard"]) for x in arms])
        relC = np.array([rel_with(a, ts, x["SR_regenerate"], 1.0) for x in arms])
        col = {k: np.array([x[k] for x in arms], float) for k in arms[0]}

        print(f"\n--- {domain} ---")
        for label, arr, key in [
            ("Revoke rate", col["revoke_rate"], "revoke"),
            ("Random-redraw success", col["random_redraw_success_rate"], "redraw"),
            ("Governed regeneration", col["regeneration_success_rate"], "regen"),
            ("Rel: baseline", relA, "relA"),
            ("Rel: discard", relB, "relB"),
            ("Rel: regenerate", relC, "relC"),
        ]:
            print(f"  {label:<24} {arr.mean():.3f} +/- {arr.std(ddof=1):.3f}")

        rows.append({
            "domain": domain, "n_seeds": len(SEEDS), "tightening": TIGHTENING,
            **{f"{k}_mean": col[k].mean() for k in col},
            **{f"{k}_sd": col[k].std(ddof=1) for k in col},
            "Rel_baseline_mean": relA.mean(), "Rel_baseline_sd": relA.std(ddof=1),
            "Rel_discard_mean": relB.mean(), "Rel_discard_sd": relB.std(ddof=1),
            "Rel_regenerate_mean": relC.mean(), "Rel_regenerate_sd": relC.std(ddof=1),
        })
    pd.DataFrame(rows).to_csv("../data/governance_multiseed.csv", index=False)

    # ---------------- Figure 3.6, tightening sweep ----------------
    print("\n=== Figure 3.6, Rel gap (regenerate minus discard) across tightening ===")
    print(f"{'tightening':>11} {'RLV gap':>9} {'HC gap':>9}")
    sweep = []
    for t in TIGHTEN_GRID:
        line = {}
        for domain, config in CONFIG:
            sub = df[df.domain == domain]
            a, ts = rel.loc[domain, "A_triad"], rel.loc[domain, "TS_mean"]
            gaps = []
            for s in SEEDS:
                x = one_arm(sub, config, t, s)
                gaps.append(rel_with(a, ts, x["SR_regenerate"], 1.0)
                            - rel_with(a, ts, x["SR_discard"], x["GC_discard"]))
            line[domain] = (float(np.mean(gaps)), float(np.std(gaps, ddof=1)))
            sweep.append({"domain": domain, "tightening": t,
                          "rel_gap_mean": line[domain][0], "rel_gap_sd": line[domain][1],
                          "reference_point": t == TIGHTENING})
        print(f"{t:>11.3f} {line['RLV'][0]:>9.3f} {line['Healthcare'][0]:>9.3f}")
    pd.DataFrame(sweep).to_csv("../data/sweep_tightening.csv", index=False)

    print("\nSaved: data/governance_multiseed.csv, data/sweep_tightening.csv")


if __name__ == "__main__":
    main()
