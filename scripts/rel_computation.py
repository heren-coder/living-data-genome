"""
Living Data Genome — Day 3-4
Compute the four Rel components (Section 2.5.6 / 2.7) from the Day 1-2
synthetic trajectories, and a simple supervised baseline for comparison.

Design notes:
  - A_triad: softened triadic alignment, Eq. (A2), instantiated through
    linear CKA as the protocol-approved similarity operator,
    applied across three views: genesis (t=0, DNA-level internal state),
    repair (t=2, RNA-level expressed state), and the declared context
    regime mean (case-facing / protocol-declared claim). This reuses a
    method the paper already commits to, rather than inventing an ad hoc
    metric for this experiment.
  - TS: Eq. (A3), exponentially bounded quadratic displacement between
    consecutive lifecycle stages, weighted by the per-gene quality metadata
    q (M_{i,t} = diag(w_g), w_g proportional to q).
  - SR: Eq. (A5)-style admissibility survival, measured at the repair
    stage (t=2) -- the stage nearest to expression/release (Table 2.2).
  - GC: governance coherence is intentionally left at a neutral placeholder
    (GC=1.0) in this script. No certified governance events (contest /
    revoke / regenerate) exist yet in the data -- those are introduced in
    the Day 5 governance-sequence demo, at which point GC becomes a
    non-trivial, measured quantity. This is stated explicitly rather than
    silently assumed.
  - Baseline: a plain logistic-regression classifier trained to predict
    repair-stage (t=2) admissibility from ONLY the genesis-stage (t=0)
    encoding -- i.e., "can a standard supervised model predict eventual
    admissibility without the governed mutation+repair correction cycle?"
    This is deliberately framed as a different epistemic question from SR
    (prediction vs. construction), consistent with the paper's own framing
    in Section 1.1-1.3, and deliberately avoids AUC as the headline metric.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED

REL_WEIGHTS = dict(A_triad=0.25, TS=0.25, SR=0.25, GC=0.25)


# ---------------------------------------------------------------------------
# Linear CKA (Kornblith et al., 2019) -- already cited in Section 2.5.6
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(num / den) if den > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# A_triad: softened triadic alignment (Eq. A2), applied to
# {genesis, repair, declared context mean} instead of {D,R,C} sensor layers
# ---------------------------------------------------------------------------

def compute_a_triad(df: pd.DataFrame, config, case_facing_noise_std: float = 0.16, seed: int = RNG_SEED + 7) -> dict:
    """
    A_triad across three independently-perturbed views, computed at the
    PER-CANDIDATE level (not averaged within each event's genesis family):
    averaging across the K-candidate family before computing CKA was found
    to artificially inflate genesis-repair alignment (~0.98 vs ~0.93 raw,
    since averaging smooths out mutation-stage noise) -- this function uses
    the raw per-candidate trajectories throughout to avoid that artifact.

      Psi_ER (evidentiary/DNA)      = genesis (t=0) gene coordinate, per candidate
      Psi_DG (operator/RNA)         = repair (t=2) gene coordinate, per candidate
      Psi_RLV (case-facing/claim)   = an independently noised reconstruction of
                                       the declared context regime mean, ONE
                                       per event, broadcast across that
                                       event's candidates (the institutional
                                       narrative is a per-event claim, not a
                                       per-candidate one)

    The case-facing view is deliberately NOT the exact protocol-declared
    regime mean: an institutional case narrative is itself a reconstruction,
    not a direct readout of the underlying regime. Its independent noise
    source decouples it from the genesis->mutation->repair chain, so A_triad
    reflects genuinely measured cross-view alignment.
    """
    rng = np.random.default_rng(seed)
    piv0 = df[df.stage_t == 0].set_index(["event_id", "candidate_id"])[["g_S", "g_A", "g_D", "g_E"]]
    piv2 = df[df.stage_t == 2].set_index(["event_id", "candidate_id"])[["g_S", "g_A", "g_D", "g_E"]]
    common = piv0.index.intersection(piv2.index)
    genesis = piv0.loc[common].values
    repair = piv2.loc[common].values

    event_ids = common.get_level_values("event_id")
    ctx_by_event = df[df.stage_t == 0].groupby("event_id")["context_label"].first()
    unique_events = ctx_by_event.index
    case_facing_by_event = {
        ev: np.clip(config.context_means[ctx_by_event[ev]] + rng.normal(0, case_facing_noise_std, size=4), 0.0, 1.0)
        for ev in unique_events
    }
    case_facing = np.array([case_facing_by_event[ev] for ev in event_ids])

    cka_gr = linear_cka(genesis, repair)
    cka_gc = linear_cka(genesis, case_facing)
    cka_rc = linear_cka(repair, case_facing)
    a_triad = (cka_gr + cka_gc + cka_rc) / 3.0
    return {"A_triad": a_triad, "CKA_genesis_repair": cka_gr,
            "CKA_genesis_context": cka_gc, "CKA_repair_context": cka_rc}


# ---------------------------------------------------------------------------
# TS: Eq. (A3) exponentially bounded quadratic displacement
# ---------------------------------------------------------------------------

def transition_stability(g_from: np.ndarray, g_to: np.ndarray, q: np.ndarray, scale: float = 8.0) -> float:
    w = q / (q.sum() + 1e-9) * scale  # normalize then scale so the exponent is discriminative
    delta = g_to - g_from
    quad = float(delta @ (w * delta))
    return float(np.exp(-quad))


def compute_ts_per_event(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_id, cand_id), grp in df.groupby(["event_id", "candidate_id"]):
        grp = grp.sort_values("stage_t")
        if len(grp) != 3:
            continue
        g = grp[["g_S", "g_A", "g_D", "g_E"]].values
        q = grp[["q_S", "q_A", "q_D", "q_E"]].values[0]  # quality metadata is event-level, constant across stages
        ts_01 = transition_stability(g[0], g[1], q)
        ts_12 = transition_stability(g[1], g[2], q)
        rows.append({"event_id": event_id, "candidate_id": cand_id,
                      "TS_genesis_to_mutation": ts_01, "TS_mutation_to_repair": ts_12,
                      "TS_event": (ts_01 + ts_12) / 2.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SR: admissibility survival at the repair stage (t=2)
# ---------------------------------------------------------------------------

def compute_sr_per_event(df: pd.DataFrame) -> pd.Series:
    repair = df[df.stage_t == 2]
    return repair.groupby("event_id")["A_g"].mean().rename("SR_event")


# ---------------------------------------------------------------------------
# Baseline: logistic regression, genesis-only features -> repair admissibility
# ---------------------------------------------------------------------------

def compute_cross_layer(df: pd.DataFrame, traces: pd.DataFrame, config,
                        case_facing_noise_std: float = 0.16,
                        seed: int = RNG_SEED + 7) -> dict:
    """
    Cross-layer coherence proxy (Eq. 2.25), two components.

      Z^(DNA)   = the protected trace u itself, raw_trace_dim-dimensional,
                  broadcast from its event to that event's candidates.
                  NOT the genesis gene coordinate: using the trace is what
                  distinguishes this diagnostic from A_triad (Eq. A2),
                  where all three views already live in gene space.
      Z^(RNA)   = repair-stage (t=2) gene coordinate carried by the released
                  package, per candidate.
      Z^(Case)  = the same institutional reconstruction used by compute_a_triad,
                  drawn under the same seed so the two diagnostics differ in
                  their DNA-level view alone.

    CL_lev is the geometric mean of the three pairwise alignments, replacing
    the arithmetic mean: an arithmetic aggregation is the l1-coherence reading
    of the density-matrix analogue and is level-only, so it cannot see how the
    three alignments are distributed. CL_bal is the min/max ratio and recovers
    exactly that information. Linear CKA admits the differing dimensionality
    of Z^(DNA), which is what lets the trace enter without being compressed.
    """
    rng = np.random.default_rng(seed)
    piv2 = df[df.stage_t == 2].set_index(["event_id", "candidate_id"])[["g_S", "g_A", "g_D", "g_E"]]
    piv0 = df[df.stage_t == 0].set_index(["event_id", "candidate_id"])[["g_S", "g_A", "g_D", "g_E"]]
    common = piv0.index.intersection(piv2.index)
    repair = piv2.loc[common].values
    event_ids = common.get_level_values("event_id")

    ucols = [c for c in traces.columns if c.startswith("u_")]
    u_by_event = traces.set_index("event_id")[ucols]
    Z_dna = u_by_event.loc[event_ids].values

    ctx_by_event = df[df.stage_t == 0].groupby("event_id")["context_label"].first()
    case_by_event = {
        ev: np.clip(config.context_means[ctx_by_event[ev]] + rng.normal(0, case_facing_noise_std, size=4), 0.0, 1.0)
        for ev in ctx_by_event.index
    }
    Z_case = np.array([case_by_event[ev] for ev in event_ids])

    a = np.array([linear_cka(Z_dna, repair),      # DNA - RNA
                  linear_cka(repair, Z_case),     # RNA - Case
                  linear_cka(Z_dna, Z_case)])     # DNA - Case
    cl_lev = float(np.prod(a) ** (1.0 / 3.0))
    cl_bal = float(a.min() / a.max())
    return {"CL_lev": cl_lev, "CL_bal": cl_bal,
            "CKA_dna_rna": float(a[0]), "CKA_rna_case": float(a[1]), "CKA_dna_case": float(a[2])}


def run_baseline(df: pd.DataFrame, seed: int) -> dict:
    genesis = df[df.stage_t == 0][["event_id", "candidate_id", "g_S", "g_A", "g_D", "g_E",
                                    "q_S", "q_A", "q_D", "q_E"]]
    repair_label = df[df.stage_t == 2][["event_id", "candidate_id", "A_g"]].rename(columns={"A_g": "target"})
    merged = genesis.merge(repair_label, on=["event_id", "candidate_id"])

    X = merged[["g_S", "g_A", "g_D", "g_E", "q_S", "q_A", "q_D", "q_E"]].values
    y = merged["target"].astype(int).values

    if y.sum() == 0 or y.sum() == len(y):
        return {"accuracy": None, "f1": None, "note": "degenerate label distribution"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # class_weight='balanced' is essential here: positive (admissible) rate is
    # ~80-86%, and a plain LogisticRegression collapses to predicting the
    # majority class (accuracy == positive rate, balanced accuracy == 0.5).
    # Balanced accuracy is reported as the headline metric for this reason.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "n_train": len(y_train), "n_test": len(y_test),
        "positive_rate_test": y_test.mean(),
    }


# ---------------------------------------------------------------------------
# Rel = A_triad^0.25 * TS^0.25 * SR^0.25 * GC^0.25
# ---------------------------------------------------------------------------

def compute_rel(domain_name: str, df: pd.DataFrame, config, gc_placeholder: float = 1.0, seed: int = RNG_SEED,
                 case_facing_noise_std: float = 0.16):
    a_triad_info = compute_a_triad(df, config, case_facing_noise_std=case_facing_noise_std, seed=seed + 7)
    ts_df = compute_ts_per_event(df)
    sr_series = compute_sr_per_event(df)

    per_event = ts_df.groupby("event_id")["TS_event"].mean().to_frame()
    per_event = per_event.join(sr_series, how="inner")
    per_event["A_triad"] = a_triad_info["A_triad"]
    per_event["GC"] = gc_placeholder

    # weighted geometric mean, Eq. (2.17)
    per_event["Rel"] = (
        per_event["A_triad"] ** REL_WEIGHTS["A_triad"]
        * per_event["TS_event"] ** REL_WEIGHTS["TS"]
        * per_event["SR_event"].clip(lower=1e-6) ** REL_WEIGHTS["SR"]
        * per_event["GC"] ** REL_WEIGHTS["GC"]
    )

    baseline_info = run_baseline(df, seed=seed)

    summary = {
        "domain": domain_name,
        "A_triad": a_triad_info["A_triad"],
        "CKA_genesis_repair": a_triad_info["CKA_genesis_repair"],
        "CKA_genesis_context": a_triad_info["CKA_genesis_context"],
        "CKA_repair_context": a_triad_info["CKA_repair_context"],
        "TS_mean": per_event["TS_event"].mean(),
        "TS_genesis_to_mutation_mean": ts_df["TS_genesis_to_mutation"].mean(),
        "TS_mutation_to_repair_mean": ts_df["TS_mutation_to_repair"].mean(),
        "SR_mean": per_event["SR_event"].mean(),
        "GC_placeholder": gc_placeholder,
        "Rel_mean": per_event["Rel"].mean(),
        "Rel_std": per_event["Rel"].std(),
        "baseline_accuracy": baseline_info.get("accuracy"),
        "baseline_balanced_accuracy": baseline_info.get("balanced_accuracy"),
        "baseline_f1": baseline_info.get("f1"),
    }
    return summary, per_event, baseline_info


if __name__ == "__main__":
    DATA = "../data/"
    df = pd.read_csv(DATA + "scenarios.csv")
    traces = pd.read_csv(DATA + "raw_traces.csv")
    cross_rows = []

    results = []
    per_event_frames = {}
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        summary, per_event, baseline_info = compute_rel(name, sub, config, seed=RNG_SEED)
        results.append(summary)
        cl = compute_cross_layer(sub, traces[traces.domain == name], config, seed=RNG_SEED + 7)
        cl["domain"] = name
        cl["tau_lev"], cl["tau_bal"] = 0.70, 0.80
        cl["Coh"] = int(cl["CL_lev"] >= 0.70 and cl["CL_bal"] >= 0.80)
        cross_rows.append(cl)
        per_event_frames[name] = per_event
        print(f"\n=== {name} ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"  baseline details: {baseline_info}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA + "rel_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(DATA + "cross_layer_coherence.csv", index=False)
    for name, pe in per_event_frames.items():
        pe.to_csv(DATA + f"rel_per_event_{name.lower()}.csv")

    print("\nSaved: data/rel_summary.csv, data/rel_per_event_{rlv,healthcare}.csv, data/cross_layer_coherence.csv")
