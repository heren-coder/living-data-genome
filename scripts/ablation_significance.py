"""
Living Data Genome -- statistical significance for ablation results.

The ablation table (Table 4.2) reports mean +/- std over 5 seeds for
A_triad and baseline balanced accuracy under each gene-dropped condition,
but never tests whether these differences from the full model exceed what
5-seed sampling noise alone would produce. This closes that gap with a
Welch's t-test (unequal variance) per condition per metric.
"""

import numpy as np
import pandas as pd
from scipy import stats

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import compute_a_triad, run_baseline
from day5_ablation import compute_a_triad_ablated, run_baseline_ablated, GENES

N_SEEDS = 5
SEED_STRIDE = 100


def full_model_samples(df: pd.DataFrame, config, seeds) -> dict:
    a_triad_samples = [compute_a_triad(df, config, case_facing_noise_std=0.16, seed=s + 7)["A_triad"] for s in seeds]
    baseline_samples = [run_baseline(df, seed=s)["balanced_accuracy"] for s in seeds]
    return {"A_triad": a_triad_samples, "baseline": baseline_samples}


def ablated_samples(df: pd.DataFrame, config, gene: str, seeds) -> dict:
    a_triad_samples = [compute_a_triad_ablated(df, config, drop=gene, seed=s + 7) for s in seeds]
    baseline_samples = [run_baseline_ablated(df, drop=gene, seed=s)["balanced_accuracy"] for s in seeds]
    return {"A_triad": a_triad_samples, "baseline": baseline_samples}


if __name__ == "__main__":
    df = pd.read_csv("../data/scenarios.csv")
    seeds = [RNG_SEED + i * SEED_STRIDE for i in range(N_SEEDS)]

    rows = []
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        full = full_model_samples(sub, config, seeds)

        for gene in GENES:
            ablated = ablated_samples(sub, config, gene, seeds)

            for metric in ["A_triad", "baseline"]:
                full_vals = np.array(full[metric])
                abl_vals = np.array(ablated[metric])
                t_stat, p_val = stats.ttest_ind(full_vals, abl_vals, equal_var=False)
                mean_diff = abl_vals.mean() - full_vals.mean()
                rows.append({
                    "domain": name, "gene_dropped": gene, "metric": metric,
                    "full_mean": full_vals.mean(), "full_std": full_vals.std(),
                    "ablated_mean": abl_vals.mean(), "ablated_std": abl_vals.std(),
                    "mean_diff": mean_diff, "t_stat": t_stat, "p_value": p_val,
                    "significant_p05": p_val < 0.05,
                })

    result_df = pd.DataFrame(rows)
    result_df.to_csv("../data/ablation_significance.csv", index=False)

    print("=== Statistical significance of ablation effects (Welch's t-test, n=5 seeds) ===\n")
    for name in ["RLV", "Healthcare"]:
        print(f"--- {name} ---")
        sub = result_df[result_df.domain == name]
        for metric in ["A_triad", "baseline"]:
            print(f"\n  {metric}:")
            msub = sub[sub.metric == metric]
            for _, r in msub.iterrows():
                sig = "***" if r["significant_p05"] else "   "
                print(f"    {r['gene_dropped']}: diff={r['mean_diff']:+.4f}  p={r['p_value']:.4f} {sig}")
        print()

    print("Saved: ../data/ablation_significance.csv")
