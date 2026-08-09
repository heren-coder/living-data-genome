"""
Living Data Genome -- Day 5, part 3: governance sequence demonstration.

Three-arm design, per the decision log:
  A) Baseline (generation 1)  -- repair-stage admissible artifacts, no
     institutional update applied.
  B) Discard (counterfactual) -- a tightened constraint regime C+ is applied
     (an independently-motivated stricter threshold, not reverse-engineered
     to hit a target revoke rate); artifacts that become inadmissible under
     C+ are REVOKED and then silently dropped, with no further lineage
     event. This is the naive "delete and forget" baseline the paper
     explicitly argues against (Section 2.5.5: "Deletion is not
     governance; it destroys auditability").
  C) Regenerate (this framework's mechanism) -- revoked artifacts are
     carried through a further mutation+repair cycle UNDER the new C+
     regime, with an explicit regenerate event linked back to the revoked
     artifact's lineage (Eq. 2.52-2.63).

Governance coherence (GC) is operationalized directly from Cons(L)
(Section 2.6): a revoked artifact with NO subsequent regenerate or explicit
terminal-disposition event is a Cons(L) violation (silent erasure). This is
not a cherry-picked penalty -- it is the literal thing the paper's lineage
discipline is designed to prevent, so testing it here is testing the
paper's own central claim, not a side effect.
"""

import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, TraceToGeneEncoder, RNG_SEED
from rel_computation import REL_WEIGHTS

TIGHTENING = 0.05  # independently-motivated stricter regime: tolerance - TIGHTENING


def tightened_bounds(config, context, tightening=TIGHTENING):
    base = config.feasibility_bounds(context)
    mean = config.context_means[context]
    out = {}
    for i, k in enumerate(["S", "A", "D", "E"]):
        tol = (base[k][1] - base[k][0]) / 2.0 - tightening
        tol = max(tol, 0.01)
        lo = max(0.0, mean[i] - tol)
        hi = min(1.0, mean[i] + tol)
        out[k] = (lo, hi)
    return out


def is_admissible(g, bounds):
    return all(bounds[k][0] <= g[i] <= bounds[k][1] for i, k in enumerate(["S", "A", "D", "E"]))


def run_governance_demo(domain_name: str, df: pd.DataFrame, config, seed: int = RNG_SEED,
                         tightening: float = TIGHTENING) -> dict:
    rng = np.random.default_rng(seed + 999)
    encoder = TraceToGeneEncoder(config.raw_trace_dim, config.context_categories, seed=seed)

    # generation 1: one representative admissible repair-stage artifact per event
    repair = df[df.stage_t == 2]
    gen1 = (
        repair[repair.A_g]
        .sort_values("candidate_id")
        .groupby("event_id")
        .first()
        .reset_index()
    )

    n_gen1 = len(gen1)
    revoked_rows = []
    kept_rows = []
    for _, row in gen1.iterrows():
        context = row["context_label"]
        g = np.array([row["g_S"], row["g_A"], row["g_D"], row["g_E"]])
        c_plus = tightened_bounds(config, context, tightening=tightening)
        if is_admissible(g, c_plus):
            kept_rows.append(row)
        else:
            revoked_rows.append(row)

    n_revoked = len(revoked_rows)
    n_kept = len(kept_rows)

    # --- Arm B: discard -- revoked artifacts dropped, no further event ---
    sr_discard = n_kept / n_gen1
    gc_discard = 1.0 - (n_revoked / n_gen1)  # every revoke-without-disposition is a Cons(L) violation

    # --- Arm C: regenerate -- revoked artifacts undergo mutation+repair under C+ ---
    regen_success = 0
    for row in revoked_rows:
        context = row["context_label"]
        c_plus = tightened_bounds(config, context, tightening=tightening)
        g0 = np.array([row["g_S"], row["g_A"], row["g_D"], row["g_E"]])
        mutated = np.clip(g0 + rng.normal(0, 0.14, size=4), 0.0, 1.0)
        region_center = np.array([(c_plus[k][0] + c_plus[k][1]) / 2 for k in ["S", "A", "D", "E"]])
        repaired = np.clip(mutated + 0.55 * (region_center - mutated) + rng.normal(0, 0.042, size=4), 0.0, 1.0)
        if is_admissible(repaired, c_plus):
            regen_success += 1

    n_regenerated_ok = regen_success
    sr_regenerate = (n_kept + n_regenerated_ok) / n_gen1
    # every revoked artifact now has an explicit regenerate event (successful or not);
    # Cons(L) is satisfied as long as the event is recorded, independent of the
    # regenerated candidate's own admissibility outcome
    gc_regenerate = 1.0

    return {
        "domain": domain_name,
        "n_gen1": n_gen1,
        "n_kept_under_Cplus": n_kept,
        "n_revoked": n_revoked,
        "revoke_rate": n_revoked / n_gen1,
        "n_regenerated_admissible": n_regenerated_ok,
        "regeneration_success_rate": n_regenerated_ok / n_revoked if n_revoked else float("nan"),
        "SR_baseline_gen1": 1.0,  # by construction, all gen1 rows are repair-admissible under the OLD regime
        "SR_discard": sr_discard,
        "SR_regenerate": sr_regenerate,
        "GC_baseline_gen1": 1.0,
        "GC_discard": gc_discard,
        "GC_regenerate": gc_regenerate,
    }


def rel_with(a_triad, ts, sr, gc):
    return (
        a_triad ** REL_WEIGHTS["A_triad"]
        * ts ** REL_WEIGHTS["TS"]
        * max(sr, 1e-6) ** REL_WEIGHTS["SR"]
        * max(gc, 1e-6) ** REL_WEIGHTS["GC"]
    )


if __name__ == "__main__":
    df = pd.read_csv("../data/scenarios.csv")
    rel_summary = pd.read_csv("../data/rel_summary.csv").set_index("domain")

    rows = []
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        demo = run_governance_demo(name, sub, config, seed=RNG_SEED)

        a_triad = rel_summary.loc[name, "A_triad"]
        ts = rel_summary.loc[name, "TS_mean"]

        rel_a = rel_with(a_triad, ts, demo["SR_baseline_gen1"], demo["GC_baseline_gen1"])
        rel_b = rel_with(a_triad, ts, demo["SR_discard"], demo["GC_discard"])
        rel_c = rel_with(a_triad, ts, demo["SR_regenerate"], demo["GC_regenerate"])

        print(f"\n=== {name} ===")
        for k, v in demo.items():
            print(f"  {k}: {v}")
        print(f"  Rel[A baseline gen1]   = {rel_a:.4f}")
        print(f"  Rel[B discard]         = {rel_b:.4f}")
        print(f"  Rel[C regenerate]      = {rel_c:.4f}")

        rows.append({
            "domain": name, **demo,
            "Rel_A_baseline": rel_a, "Rel_B_discard": rel_b, "Rel_C_regenerate": rel_c,
        })

    pd.DataFrame(rows).to_csv("../data/governance_demo_results.csv", index=False)
    print("\nSaved: ../data/governance_demo_results.csv")
