"""
Living Data Genome -- Section 2.5.7 operationalization: context-mismatch
(ectopic expression) screening.

Uses infrastructure already built in Day 1-2 (context_label, pi_lookup.csv)
that was deliberately included from the start for exactly this purpose.

  rho_{i,t}      = D(g_{i,t})            gene-dominance profile (simplex)
  Xi_{i,t}(c)    = JSD(rho, pi(c))       observed-vs-expected divergence
  A_pi(i,t;c)    = 1[Xi <= xi]           context-mismatch admissibility
  Route          = proceed / quarantine-repair / inadmissible   (Eq. 2.20)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, _simplex


def compute_xi_and_route(df: pd.DataFrame, pi_lookup: pd.DataFrame, domain: str, xi_threshold: float) -> pd.DataFrame:
    pi_map = {
        row["context_label"]: np.array([row["pi_S"], row["pi_A"], row["pi_D"], row["pi_E"]])
        for _, row in pi_lookup[pi_lookup.domain == domain].iterrows()
    }

    repair = df[df.stage_t == 2].copy()
    rho = repair[["g_S", "g_A", "g_D", "g_E"]].apply(lambda r: _simplex(r.values), axis=1)
    repair["Xi"] = [
        jensenshannon(rho.iloc[i], pi_map[repair.iloc[i]["context_label"]]) ** 2  # JSD^2 = JS divergence
        for i in range(len(repair))
    ]
    repair["A_pi"] = (repair["Xi"] <= xi_threshold).astype(int)

    def route(row):
        if not row["A_g"]:
            return "inadmissible"
        elif row["A_pi"]:
            return "proceed"
        else:
            return "quarantine_repair"

    repair["route"] = repair.apply(route, axis=1)
    return repair


def context_repair(g: np.ndarray, pi_c: np.ndarray, bounds: dict, pull_fraction: float = 0.4) -> np.ndarray:
    """
    Repair operator for context-mismatch (Section 2.5.3's T_rep, applied to
    the dominance profile rather than raw admissibility): pulls the
    candidate's gene-dominance profile toward pi(c) by pull_fraction while
    preserving its total magnitude, then re-clips to the admissible bounds.
    This is the mechanism Section 2.5.7 promises but does not itself
    simulate: quarantine is not a dead end, it routes back through repair.
    """
    total_mass = g.sum()
    rho = _simplex(g)
    new_rho = rho + pull_fraction * (pi_c - rho)
    new_rho = new_rho / new_rho.sum()  # renormalize after the pull
    new_g = new_rho * total_mass
    for i, k in enumerate(["S", "A", "D", "E"]):
        new_g[i] = np.clip(new_g[i], bounds[k][0], bounds[k][1])
    return new_g


def run_context_repair_loop(df: pd.DataFrame, pi_lookup: pd.DataFrame, domain: str, config,
                             xi_threshold: float = 0.030, pull_fraction: float = 0.4) -> dict:
    pi_map = {
        row["context_label"]: np.array([row["pi_S"], row["pi_A"], row["pi_D"], row["pi_E"]])
        for _, row in pi_lookup[pi_lookup.domain == domain].iterrows()
    }
    routed = compute_xi_and_route(df, pi_lookup, domain, xi_threshold)
    quarantined = routed[routed.route == "quarantine_repair"].copy()

    xi_before = quarantined["Xi"].values.copy()
    xi_after = []
    still_admissible = []
    for _, row in quarantined.iterrows():
        context = row["context_label"]
        g = np.array([row["g_S"], row["g_A"], row["g_D"], row["g_E"]])
        bounds = config.feasibility_bounds(context)
        g_repaired = context_repair(g, pi_map[context], bounds, pull_fraction)
        rho_repaired = _simplex(g_repaired)
        xi_new = jensenshannon(rho_repaired, pi_map[context]) ** 2
        xi_after.append(xi_new)
        still_admissible.append(all(bounds[k][0] <= g_repaired[i] <= bounds[k][1]
                                     for i, k in enumerate(["S", "A", "D", "E"])))

    xi_after = np.array(xi_after)
    resolved = (xi_after <= xi_threshold)
    return {
        "domain": domain,
        "n_quarantined": len(quarantined),
        "Xi_mean_before": xi_before.mean(),
        "Xi_mean_after": xi_after.mean(),
        "Xi_reduction_pct": 100 * (1 - xi_after.mean() / xi_before.mean()),
        "resolved_rate": resolved.mean(),
        "still_admissible_rate": np.mean(still_admissible),
    }


if __name__ == "__main__":



    df = pd.read_csv("../data/scenarios.csv")
    pi_lookup = pd.read_csv("../data/pi_lookup.csv")

    # first inspect the Xi distribution to choose a defensible threshold, then sweep it
    print("=== Xi distribution (xi_threshold=0.10 for inspection) ===")
    for domain in ["RLV", "Healthcare"]:
        sub = df[df.domain == domain]
        res = compute_xi_and_route(sub, pi_lookup, domain, xi_threshold=0.10)
        print(f"\n{domain}: Xi mean={res['Xi'].mean():.4f}, median={res['Xi'].median():.4f}, "
              f"90th pct={res['Xi'].quantile(0.9):.4f}, max={res['Xi'].max():.4f}")

    print("\n=== Threshold sensitivity sweep ===")
    rows = []
    for xi_t in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        for domain in ["RLV", "Healthcare"]:
            sub = df[df.domain == domain]
            res = compute_xi_and_route(sub, pi_lookup, domain, xi_threshold=xi_t)
            counts = res["route"].value_counts(normalize=True)
            rows.append({
                "xi_threshold": xi_t, "domain": domain,
                "proceed": counts.get("proceed", 0.0),
                "quarantine_repair": counts.get("quarantine_repair", 0.0),
                "inadmissible": counts.get("inadmissible", 0.0),
            })
    sweep_df = pd.DataFrame(rows)
    print(sweep_df.round(3).to_string(index=False))
    sweep_df.to_csv("../data/ectopic_expression_sweep.csv", index=False)

    # reference operating point
    print("\n=== Reference operating point (xi_threshold=0.10) ===")
    ref_rows = []
    for domain in ["RLV", "Healthcare"]:
        sub = df[df.domain == domain]
        res = compute_xi_and_route(sub, pi_lookup, domain, xi_threshold=0.10)
        counts = res["route"].value_counts(normalize=True)
        ref_rows.append({
            "domain": domain,
            "proceed": counts.get("proceed", 0.0),
            "quarantine_repair": counts.get("quarantine_repair", 0.0),
            "inadmissible": counts.get("inadmissible", 0.0),
            "Xi_mean": res["Xi"].mean(),
        })
        print(domain, ref_rows[-1])
    pd.DataFrame(ref_rows).to_csv("../data/ectopic_expression_reference.csv", index=False)
    print("\nSaved: data/ectopic_expression_sweep.csv, data/ectopic_expression_reference.csv")
