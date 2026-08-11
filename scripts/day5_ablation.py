"""
Living Data Genome -- Day 5, part 1: gene ablation.

For each of the four genes {S, A, D, E} in turn, the gene is "ablated"
(dropped from the representation / admissibility test / classifier
features) and the four reliability components are recomputed on the
remaining three dimensions.

Design note on interpretation:
  Dropping a gene from the admissibility test G(C) REMOVES a constraint,
  so SR can only stay the same or INCREASE when a gene is ablated -- this
  is expected and reported honestly, not treated as a "problem" to hide.
  The informative signal is elsewhere: A_triad (representational alignment)
  and the baseline classifier's balanced accuracy are expected to DEGRADE
  when a genuinely load-bearing gene is dropped, even as SR misleadingly
  looks better. This is precisely why Rel is defined as a multi-component,
  weakest-link score rather than admissibility rate alone: a single-metric
  view (SR) could be gamed by shrinking the constraint set, while the
  fuller decomposition exposes the resulting loss of coherence.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import linear_cka, transition_stability, REL_WEIGHTS

GENES = ["S", "A", "D", "E"]


def gene_cols(drop: str = None):
    cols = [g for g in GENES if g != drop]
    return [f"g_{g}" for g in cols], [f"q_{g}" for g in cols]


def compute_a_triad_ablated(df: pd.DataFrame, config, drop: str, case_facing_noise_std: float = 0.16,
                             seed: int = RNG_SEED + 7) -> float:
    gcols, _ = gene_cols(drop)
    rng = np.random.default_rng(seed)
    piv0 = df[df.stage_t == 0].set_index(["event_id", "candidate_id"])[gcols]
    piv2 = df[df.stage_t == 2].set_index(["event_id", "candidate_id"])[gcols]
    common = piv0.index.intersection(piv2.index)
    genesis = piv0.loc[common].values
    repair = piv2.loc[common].values

    event_ids = common.get_level_values("event_id")
    ctx_by_event = df[df.stage_t == 0].groupby("event_id")["context_label"].first()
    keep_idx = [i for i, g in enumerate(GENES) if g != drop]
    case_facing_by_event = {
        ev: np.clip(
            config.context_means[ctx_by_event[ev]][keep_idx] + rng.normal(0, case_facing_noise_std, size=len(keep_idx)),
            0.0, 1.0,
        )
        for ev in ctx_by_event.index
    }
    case_facing = np.array([case_facing_by_event[ev] for ev in event_ids])

    cka_gr = linear_cka(genesis, repair)
    cka_gc = linear_cka(genesis, case_facing)
    cka_rc = linear_cka(repair, case_facing)
    return (cka_gr + cka_gc + cka_rc) / 3.0


def compute_sr_ablated(df: pd.DataFrame, config, drop: str) -> float:
    """Recompute admissibility using only the remaining 3 gene bounds."""
    repair = df[df.stage_t == 2].copy()
    keep = [g for g in GENES if g != drop]

    def check(row):
        bounds = config.feasibility_bounds(row["context_label"])
        return all(bounds[g][0] <= row[f"g_{g}"] <= bounds[g][1] for g in keep)

    admissible = repair.apply(check, axis=1)
    return admissible.groupby(repair["event_id"]).mean().mean()


def compute_ts_ablated(df: pd.DataFrame, drop: str) -> dict:
    gcols, qcols = gene_cols(drop)
    rows_01, rows_12 = [], []
    for (event_id, cand_id), grp in df.groupby(["event_id", "candidate_id"]):
        grp = grp.sort_values("stage_t")
        if len(grp) != 3:
            continue
        g = grp[gcols].values
        q = grp[qcols].values[0]
        rows_01.append(transition_stability(g[0], g[1], q))
        rows_12.append(transition_stability(g[1], g[2], q))
    return {"TS_genesis_to_mutation": float(np.mean(rows_01)), "TS_mutation_to_repair": float(np.mean(rows_12))}


def run_baseline_ablated(df: pd.DataFrame, drop: str, seed: int) -> dict:
    gcols, qcols = gene_cols(drop)
    genesis = df[df.stage_t == 0][["event_id", "candidate_id"] + gcols + qcols]
    repair_label = df[df.stage_t == 2][["event_id", "candidate_id", "A_g"]].rename(columns={"A_g": "target"})
    merged = genesis.merge(repair_label, on=["event_id", "candidate_id"])

    X = merged[gcols + qcols].values
    y = merged["target"].astype(int).values
    if y.sum() == 0 or y.sum() == len(y):
        return {"balanced_accuracy": None, "f1": None}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {"balanced_accuracy": balanced_accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}


def run_ablation(domain_name: str, df: pd.DataFrame, config, full_rel: float, gc_placeholder: float = 1.0,
                  seed: int = RNG_SEED, n_seeds: int = 5, seed_stride: int = 100) -> pd.DataFrame:
    """
    A_triad and the baseline classifier are each averaged over n_seeds
    independent seeds. A single-seed pilot run showed that some ablation
    deltas (especially for A_triad's case-facing noise draw, and the
    baseline's train/test split) were within noise of each other and could
    even flip which gene appeared most load-bearing between a single-seed
    estimate and a 5-seed average -- so multi-seed averaging is treated as
    the default here, not an optional robustness check.
    SR is deterministic given the data (no stochastic component in its
    definition) and TS depends only on fixed generated trajectories, so
    both are computed once per gene rather than re-sampled.
    """
    seeds = [seed + i * seed_stride for i in range(n_seeds)]
    rows = []
    for gene in GENES:
        a_triad_samples = [compute_a_triad_ablated(df, config, drop=gene, seed=s + 7) for s in seeds]
        baseline_samples = [run_baseline_ablated(df, drop=gene, seed=s) for s in seeds]
        ba_samples = [b["balanced_accuracy"] for b in baseline_samples if b["balanced_accuracy"] is not None]
        f1_samples = [b["f1"] for b in baseline_samples if b["f1"] is not None]

        a_triad = float(np.mean(a_triad_samples))
        a_triad_std = float(np.std(a_triad_samples))
        sr = compute_sr_ablated(df, config, drop=gene)
        ts_info = compute_ts_ablated(df, drop=gene)
        ts_mean = (ts_info["TS_genesis_to_mutation"] + ts_info["TS_mutation_to_repair"]) / 2.0

        rel_ablated = (
            a_triad ** REL_WEIGHTS["A_triad"]
            * ts_mean ** REL_WEIGHTS["TS"]
            * max(sr, 1e-6) ** REL_WEIGHTS["SR"]
            * gc_placeholder ** REL_WEIGHTS["GC"]
        )

        rows.append({
            "domain": domain_name,
            "gene_dropped": gene,
            "A_triad_mean": a_triad,
            "A_triad_std": a_triad_std,
            "TS_mean": ts_mean,
            "SR": sr,
            "Rel": rel_ablated,
            "Rel_delta_vs_full": rel_ablated - full_rel,
            "baseline_balanced_accuracy_mean": float(np.mean(ba_samples)),
            "baseline_balanced_accuracy_std": float(np.std(ba_samples)),
            "baseline_f1_mean": float(np.mean(f1_samples)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("../data/scenarios.csv")
    rel_summary = pd.read_csv("../data/rel_summary.csv").set_index("domain")

    all_results = []
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        full_rel = rel_summary.loc[name, "Rel_mean"]
        full_baseline_ba = rel_summary.loc[name, "baseline_balanced_accuracy"]
        ablation_df = run_ablation(name, sub, config, full_rel=full_rel, seed=RNG_SEED)
        all_results.append(ablation_df)

        print(f"\n=== {name} (full Rel={full_rel:.4f}, full baseline balanced_acc={full_baseline_ba:.4f}) ===")
        print(ablation_df.round(4).to_string(index=False))

    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("../data/ablation_results.csv", index=False)
    print("\nSaved: data/ablation_results.csv")
