"""
actx_routing.py -- Equation (2.19) A_ctx and the four-route rule of Equation (2.20).

Why this script exists
----------------------
`day5_ectopic.py` implements only three routes:

    inadmissible / proceed / quarantine_repair

The context-attribution indicator A_ctx of Equation (2.19) is never evaluated
there. As a consequence every flagged candidate was written into the
quarantine/repair row of Table 3.6, and the table note asserts that
"no candidate met the context-attribution condition at the declared separation
margin". That assertion is not a measurement, and it is false: at the
declared xi = 0.030 and eta = 0.010, the context-review route holds
39 candidates in the RLV configuration and 27 in the healthcare vignette.

This script evaluates A_ctx, produces the four-route outcome of Table 3.6, and
computes Table 3.7 on the quarantine set (the candidates routed to
context review are no longer sent through the context-repair operator, which is
exactly the behaviour Section 2.5.7 prescribes).

Inputs : ../scenarios.csv , ../pi_lookup.csv
Outputs: data/context_routing_and_repair.csv
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, _simplex
from day5_ectopic import context_repair

XI = 0.030          # mismatch tolerance, Eq. (2.18), Table 3.8
ETA = 0.010         # separation margin, Eq. (2.19), Table 3.8
PULL = 0.40         # context-repair pull fraction, shared with day6_ectopic_mislabel
GENES = ["S", "A", "D", "E"]
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]


def jsd(p, q):
    return float(jensenshannon(p, q) ** 2)


def pi_map(pi_lookup, domain):
    return {r["context_label"]: np.array([r.pi_S, r.pi_A, r.pi_D, r.pi_E])
            for _, r in pi_lookup[pi_lookup.domain == domain].iterrows()}


def score_routes(df, pi_lookup, domain, xi=XI, eta=ETA):
    """Repair-stage candidates with Xi, Xi*, A_pi, A_ctx and the four-route label."""
    P = pi_map(pi_lookup, domain)
    labels = sorted(P)
    d = df[(df.domain == domain) & (df.stage_t == 2)].copy().reset_index(drop=True)

    rho = np.vstack([_simplex(r) for r in d[["g_S", "g_A", "g_D", "g_E"]].values])
    d["Xi"] = [jsd(rho[i], P[d.context_label[i]]) for i in range(len(d))]

    allx = np.array([[jsd(rho[i], P[c]) for c in labels] for i in range(len(d))])
    d["Xi_star"] = allx.min(axis=1)
    d["c_star"] = [labels[j] for j in allx.argmin(axis=1)]

    d["A_pi"] = (d["Xi"] <= xi).astype(int)                      # Eq. (2.18)
    d["A_ctx"] = ((d["Xi"] - d["Xi_star"]) >= eta).astype(int)   # Eq. (2.19)

    def route(r):                                                # Eq. (2.20)
        if not r["A_g"]:
            return "inadmissible"
        if r["A_pi"] == 1:
            return "proceed"
        return "context_review" if r["A_ctx"] == 1 else "quarantine_repair"

    d["route"] = d.apply(route, axis=1)
    return d


def table_3_7(d, config, pi_lookup, domain, subset, xi=XI, pull=PULL):
    """Context-repair outcome for a given quarantine subset."""
    P = pi_map(pi_lookup, domain)
    before, after, still_feasible = [], [], []
    for _, r in subset.iterrows():
        g = np.array([r.g_S, r.g_A, r.g_D, r.g_E])
        bounds = config.feasibility_bounds(r.context_label)
        g2 = context_repair(g, P[r.context_label], bounds, pull)
        before.append(jsd(_simplex(g), P[r.context_label]))
        after.append(jsd(_simplex(g2), P[r.context_label]))
        still_feasible.append(all(bounds[k][0] - 1e-9 <= g2[i] <= bounds[k][1] + 1e-9
                                  for i, k in enumerate(GENES)))
    before, after = np.array(before), np.array(after)
    return {
        "n_quarantined": len(subset),
        "Xi_before": before.mean(),
        "Xi_after": after.mean(),
        "Xi_reduction": 1 - after.mean() / before.mean(),
        "resolved_below_xi": (after <= xi).mean(),
        "remains_feasible": float(np.mean(still_feasible)),
    }


def main():
    df = pd.read_csv("../data/scenarios.csv")
    pil = pd.read_csv("../data/pi_lookup.csv")

    rows = []
    for domain, config in CONFIG:
        d = score_routes(df, pil, domain)
        n = len(d)
        counts = d.route.value_counts()
        feas = d[d.A_g]
        flagged = feas[feas.A_pi == 0]

        print(f"\n=== {domain} (n={n} repair-stage candidates) ===")
        print("  corrected Table 3.6, four routes")
        for key, published in [("proceed", None), ("context_review", None),
                               ("quarantine_repair", None), ("inadmissible", None)]:
            c = int(counts.get(key, 0))
            print(f"    {key:<20} {c:>4}   {c / n:6.1%}")
        print(f"    {'total':<20} {int(counts.sum()):>4}")

        # sanity: every context-review candidate must have c* different from the declared label
        cr = d[d.route == "context_review"]
        assert (cr.c_star != cr.context_label).all(), "context review implies a different best label"

        print("  corrected Table 3.7")
        new = table_3_7(d, config, pil, domain, flagged[flagged.A_ctx == 0])
        for tag, r in [("context repair", new)]:
            print(f"    {tag:<26} n={r['n_quarantined']:>4}  "
                  f"Xi {r['Xi_before']:.4f} -> {r['Xi_after']:.4f}  "
                  f"reduction {r['Xi_reduction']:.1%}  resolved {r['resolved_below_xi']:.1%}  "
                  f"feasible {r['remains_feasible']:.0%}")

        rows.append({
            "domain": domain, "n_repair_stage": n,
            "proceed": int(counts.get("proceed", 0)),
            "context_review": int(counts.get("context_review", 0)),
            "quarantine_repair": int(counts.get("quarantine_repair", 0)),
            "inadmissible": int(counts.get("inadmissible", 0)),
            "proceed_pct": counts.get("proceed", 0) / n,
            "context_review_pct": counts.get("context_review", 0) / n,
            "quarantine_repair_pct": counts.get("quarantine_repair", 0) / n,
            "inadmissible_pct": counts.get("inadmissible", 0) / n,
            **{f"repair_{k}": v for k, v in new.items()},
        })

    out = pd.DataFrame(rows)
    out.to_csv("../data/context_routing_and_repair.csv", index=False)
    print("\nSaved: data/context_routing_and_repair.csv")


if __name__ == "__main__":
    main()
