"""
Living Data Genome -- protocol sensitivity sweeps.

Three declarations the paper promises but does not currently deliver:

  S1  feasibility tolerance delta      (Section 3 intro: "its full sensitivity
                                        sweep is reported later in this section")
  S2  reliability weighting vector w   (Section 2.5.6: "Sensitivity analyses over
                                        alternative weightings are reported in
                                        Section 3"; Discussion repeats it)
  S3  perturbation radius eps_P        (Section 2.7: "Section 3 accordingly
                                        reports a sweep over this radius rather
                                        than a single operating point")

Outputs feed Section 3.8 and Figures 3.9-3.10; the declared radius is 0.25,
recorded in Table 3.8.

Nothing here is a new estimator. Each sweep varies exactly one protocol-declared
constant and recomputes the SAME components defined in rel_computation.py.

S1 note: delta enters only through A_g, hence only through SR. A_triad and TS are
computed on gene coordinates, which the tolerance does not move. The sweep
therefore reports Rel(delta) with A_triad and TS held at their measured values,
and additionally reports the stage-wise admissibility curve, since the
genesis-mutation-repair dip-and-recovery pattern of Figure 3.3 is the claim most
exposed to the objection that it follows from one tolerance choice.

S3 note: the sweep is over a DECLARED radius, not over the displacements the
generator happened to produce. Perturbations are drawn uniformly from the L2 ball
of radius eps_P and applied to the repair-stage coordinate; TS is then the same
exponentially bounded quadratic form of Eq. (A3) under the same weighting
matrix M = s_TS * diag(q_normalized). The realized envelope of the generator is
reported alongside as a reference marker, not as the sweep.
"""

import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG
from rel_computation import compute_a_triad, compute_ts_per_event, transition_stability

GENES = ["S", "A", "D", "E"]
GCOLS = [f"g_{k}" for k in GENES]
QCOLS = [f"q_{k}" for k in GENES]
STABILITY_SCALE = 8.0
SEED = 42

CONFIGS = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
OPERATING_DELTA = {"RLV": 0.33, "Healthcare": 0.31}

# GC measured in the Section 3.4 governance demonstration (discard arm)
GC_DISCARD = {"RLV": 0.717172, "Healthcare": 0.736041}


def load() -> pd.DataFrame:
    return pd.read_csv("../data/scenarios.csv")


# ---------------------------------------------------------------------------
# S1: feasibility tolerance
# ---------------------------------------------------------------------------

def admissible_at(df: pd.DataFrame, config, delta: float) -> np.ndarray:
    """Recompute A_g under a tolerance delta, using the same band rule as
    DomainConfig.feasibility_bounds (clipped to [0,1] per coordinate)."""
    means = np.array([config.context_means[c] for c in df["context_label"].values])
    lo = np.clip(means - delta, 0.0, None)
    hi = np.clip(means + delta, None, 1.0)
    g = df[GCOLS].values
    return np.all((g >= lo) & (g <= hi), axis=1)


def sweep_delta(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for domain, config in CONFIGS.items():
        d = df[df.domain == domain].copy()
        a_triad = compute_a_triad(d, config, seed=SEED + 7)["A_triad"]
        ts = compute_ts_per_event(d)["TS_event"].mean()
        for delta in grid:
            adm = admissible_at(d, config, float(delta))
            d = d.assign(_adm=adm)
            by_stage = d.groupby("stage_t")["_adm"].mean()
            sr = d[d.stage_t == 2].groupby("event_id")["_adm"].mean().mean()
            rel = (a_triad ** 0.25) * (ts ** 0.25) * (max(sr, 1e-6) ** 0.25) * (1.0 ** 0.25)
            rows.append({
                "domain": domain, "delta": round(float(delta), 4),
                "adm_genesis": by_stage.get(0, np.nan),
                "adm_mutation": by_stage.get(1, np.nan),
                "adm_repair": by_stage.get(2, np.nan),
                "dip": by_stage.get(0, np.nan) - by_stage.get(1, np.nan),
                "recovery": by_stage.get(2, np.nan) - by_stage.get(1, np.nan),
                "A_triad": a_triad, "TS": ts, "SR": sr, "Rel": rel,
                "operating_point": abs(float(delta) - OPERATING_DELTA[domain]) < 1e-9,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S2: reliability weighting vector
# ---------------------------------------------------------------------------

WEIGHTINGS = {
    "neutral":            (0.25, 0.25, 0.25, 0.25),
    "alignment_first":    (0.55, 0.15, 0.15, 0.15),
    "stability_first":    (0.15, 0.55, 0.15, 0.15),
    "admissibility_first": (0.15, 0.15, 0.55, 0.15),
    "governance_first":   (0.15, 0.15, 0.15, 0.55),
    "release_facing":     (0.30, 0.10, 0.30, 0.30),
    "supervisory_facing": (0.30, 0.30, 0.30, 0.10),
}


def sweep_weights(components: dict) -> pd.DataFrame:
    rows = []
    for domain, comp in components.items():
        for gc_label, gc in [("GC=1 (main results)", 1.0),
                             ("GC measured, discard arm", GC_DISCARD[domain])]:
            vals = np.array([comp["A_triad"], comp["TS"], comp["SR"], gc])
            for name, w in WEIGHTINGS.items():
                rel = float(np.prod(vals ** np.array(w)))
                rows.append({"domain": domain, "gc_regime": gc_label, "weighting": name,
                             "w_A": w[0], "w_TS": w[1], "w_SR": w[2], "w_GC": w[3],
                             "Rel": rel})
    out = pd.DataFrame(rows)
    spread = (out.groupby(["domain", "gc_regime"])["Rel"]
                 .agg(["min", "max", "mean"]).reset_index())
    spread["range"] = spread["max"] - spread["min"]
    return out, spread


# ---------------------------------------------------------------------------
# S3: declared perturbation radius
# ---------------------------------------------------------------------------

def sample_ball(rng, n: int, dim: int, radius: float) -> np.ndarray:
    """Uniform sample from the L2 ball of the given radius."""
    v = rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = radius * rng.random(size=(n, 1)) ** (1.0 / dim)
    return v * r


def sweep_eps(df: pd.DataFrame, grid: np.ndarray, n_draws: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for domain in CONFIGS:
        d = df[(df.domain == domain) & (df.stage_t == 2)]
        g = d[GCOLS].values
        q = d[QCOLS].values
        w = q / (q.sum(axis=1, keepdims=True) + 1e-9) * STABILITY_SCALE
        for eps in grid:
            ts_all = []
            for _ in range(n_draws):
                pert = sample_ball(rng, len(g), 4, float(eps))
                quad = np.sum(pert * (w * pert), axis=1)
                ts_all.append(np.exp(-quad))
            ts_all = np.concatenate(ts_all)
            rows.append({"domain": domain, "eps_P": round(float(eps), 4),
                         "TS_mean": ts_all.mean(), "TS_p05": np.percentile(ts_all, 5),
                         "TS_p95": np.percentile(ts_all, 95), "TS_min": ts_all.min()})
    return pd.DataFrame(rows)


def realized_envelope(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in CONFIGS:
        d = df[df.domain == domain]
        piv = {t: d[d.stage_t == t].set_index(["event_id", "candidate_id"])[GCOLS] for t in (0, 1, 2)}
        common = piv[0].index.intersection(piv[1].index).intersection(piv[2].index)
        d01 = np.linalg.norm(piv[1].loc[common].values - piv[0].loc[common].values, axis=1)
        d12 = np.linalg.norm(piv[2].loc[common].values - piv[1].loc[common].values, axis=1)
        both = np.concatenate([d01, d12])
        rows.append({"domain": domain, "n_transitions": len(both),
                     "displacement_mean": both.mean(),
                     "displacement_p95": np.percentile(both, 95),
                     "displacement_max": both.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load()

    print("=" * 72)
    print("S1  feasibility tolerance delta")
    print("=" * 72)
    delta_grid = np.round(np.arange(0.15, 0.55 + 1e-9, 0.01), 4)
    delta_grid = np.unique(np.concatenate([delta_grid, [0.31, 0.33]]))
    s1 = sweep_delta(df, delta_grid)
    s1.to_csv("../data/sweep_delta.csv", index=False)
    for domain in CONFIGS:
        sub = s1[s1.domain == domain]
        show = sub[sub.delta.isin([0.15, 0.20, 0.25, 0.30, OPERATING_DELTA[domain], 0.40, 0.45, 0.50])]
        print(f"\n{domain}")
        print(show[["delta", "adm_genesis", "adm_mutation", "adm_repair",
                    "recovery", "SR", "Rel", "operating_point"]].to_string(index=False, float_format="%.3f"))

    components = {}
    for domain, config in CONFIGS.items():
        d = df[df.domain == domain]
        components[domain] = {
            "A_triad": compute_a_triad(d, config, seed=SEED + 7)["A_triad"],
            "TS": compute_ts_per_event(d)["TS_event"].mean(),
            "SR": d[d.stage_t == 2].groupby("event_id")["A_g"].mean().mean(),
        }

    print("\n" + "=" * 72)
    print("S2  reliability weighting vector w")
    print("=" * 72)
    s2, s2_spread = sweep_weights(components)
    s2.to_csv("../data/sweep_weights.csv", index=False)
    for domain in CONFIGS:
        print(f"\n{domain}  (A_triad={components[domain]['A_triad']:.3f}, "
              f"TS={components[domain]['TS']:.3f}, SR={components[domain]['SR']:.3f})")
        print(s2[s2.domain == domain].pivot(index="weighting", columns="gc_regime",
                                            values="Rel").to_string(float_format="%.3f"))
    print("\nspread across weightings")
    print(s2_spread.to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 72)
    print("S3  declared perturbation radius eps_P")
    print("=" * 72)
    eps_grid = np.round(np.arange(0.05, 0.80 + 1e-9, 0.05), 4)
    s3 = sweep_eps(df, eps_grid)
    s3.to_csv("../data/sweep_eps.csv", index=False)
    for domain in CONFIGS:
        print(f"\n{domain}")
        print(s3[s3.domain == domain][["eps_P", "TS_mean", "TS_p05", "TS_p95"]]
              .to_string(index=False, float_format="%.3f"))
    print("\nrealized generator envelope (reference marker, NOT the sweep)")
    print(realized_envelope(df).to_string(index=False, float_format="%.3f"))
