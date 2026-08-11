"""
Living Data Genome -- statistical significance for ablation results (Table 3.3).

CHANGED IN v43
--------------
The v42 script used stats.ttest_ind (Welch, INDEPENDENT samples). The two arms
are matched by seed: full_model_samples() and ablated_samples() are evaluated on
the same seed list, so the same underlying candidate draw and the same
case-facing noise stream feed both arms. Applying an independent-samples test to
matched data discards the pairing and loses power. This version uses
stats.ttest_rel.

The change is not cosmetic. Under Holm correction across all sixteen tests:

    independent (v42):  5 of 8 marked differences survive
    paired      (v43):  7 of 8 marked differences survive

The two that newly survive are the Healthcare alignment effect of
ConcurrentRecordDensity (p 0.0078 -> 0.0016) and the Healthcare baseline effect
of AccessControlRegime (p 0.0143 -> 0.0046). The one that still fails is the RLV
baseline effect of Speed (p 0.0357).

ALSO ADDED
----------
  - Cohen's d_z (paired standardized effect size) = mean(diff) / sd(diff)
  - 95% confidence interval on the mean difference
  - Holm-corrected significance flag, computed in-script rather than by hand

INTERPRETING d_z
----------------
d_z reaches -11.6 for Density. This is not a claim of a huge practical effect.
The pipeline is deterministic given a seed; the only between-seed variation is
the case-facing noise draw, so between-seed variance is tiny and any systematic
shift standardizes to a large value. These tests establish that the ablation
effects are STABLE ACROSS SEEDS, not that they generalize beyond the two
configurations. The mean differences and their intervals are the interpretable
quantities. Table 3.3's note in v43 says exactly this.

Usage:  python3 ablation_significance.py
Needs:  scenarios.csv at the path set in SCENARIOS below, plus generator.py,
        rel_computation.py and day5_ablation.py importable from cwd.
"""

import numpy as np
import pandas as pd
from scipy import stats

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED
from rel_computation import compute_a_triad, run_baseline
from day5_ablation import compute_a_triad_ablated, run_baseline_ablated, GENES

N_SEEDS = 5
SEED_STRIDE = 100
CASE_FACING_NOISE = 0.16          # sigma, Table 3.8
ALPHA = 0.05

SCENARIOS = "../data/scenarios.csv"
OUT_CSV = "../data/ablation_significance.csv"


def full_model_samples(df: pd.DataFrame, config, seeds) -> dict:
    """Full-model A_triad and baseline balanced accuracy, one value per seed."""
    return {
        "A_triad": [compute_a_triad(df, config, case_facing_noise_std=CASE_FACING_NOISE,
                                    seed=s + 7)["A_triad"] for s in seeds],
        "baseline": [run_baseline(df, seed=s)["balanced_accuracy"] for s in seeds],
    }


def ablated_samples(df: pd.DataFrame, config, gene: str, seeds) -> dict:
    """Same two metrics with one gene dropped. Seeds match full_model_samples."""
    return {
        "A_triad": [compute_a_triad_ablated(df, config, drop=gene, seed=s + 7) for s in seeds],
        "baseline": [run_baseline_ablated(df, drop=gene, seed=s)["balanced_accuracy"] for s in seeds],
    }


def holm(pvals, alpha=ALPHA):
    """Holm-Bonferroni step-down. Returns a boolean array aligned to pvals."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    keep = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            keep[idx] = True
        else:
            break                      # step-down: stop at the first failure
    return keep


def paired_stats(full_vals, abl_vals):
    """Paired test, effect size and CI for one gene-by-metric cell."""
    full_vals = np.asarray(full_vals, dtype=float)
    abl_vals = np.asarray(abl_vals, dtype=float)
    diff = abl_vals - full_vals            # signed: negative means ablation hurt
    n = len(diff)
    t_stat, p_val = stats.ttest_rel(abl_vals, full_vals)
    sd = diff.std(ddof=1)
    d_z = diff.mean() / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return {
        "full_mean": full_vals.mean(), "full_std": full_vals.std(ddof=1),
        "ablated_mean": abl_vals.mean(), "ablated_std": abl_vals.std(ddof=1),
        "mean_diff": diff.mean(), "diff_sd": sd,
        "ci_lo": diff.mean() - crit * se, "ci_hi": diff.mean() + crit * se,
        "t_stat": t_stat, "p_value": p_val, "cohens_dz": d_z,
        "significant_p05": bool(p_val < ALPHA),
    }


if __name__ == "__main__":
    df = pd.read_csv(SCENARIOS)
    seeds = [RNG_SEED + i * SEED_STRIDE for i in range(N_SEEDS)]

    rows = []
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        full = full_model_samples(sub, config, seeds)
        for gene in GENES:
            ablated = ablated_samples(sub, config, gene, seeds)
            for metric in ["A_triad", "baseline"]:
                row = {"domain": name, "gene_dropped": gene, "metric": metric}
                row.update(paired_stats(full[metric], ablated[metric]))
                rows.append(row)

    result = pd.DataFrame(rows)
    result["holm_significant"] = holm(result["p_value"].values)
    result.to_csv(OUT_CSV, index=False)

    print("=== Ablation significance: paired t-test over 5 matched seeds ===\n")
    for name in ["RLV", "Healthcare"]:
        print(f"--- {name} ---")
        sub = result[result.domain == name]
        for metric in ["A_triad", "baseline"]:
            print(f"\n  {metric}:")
            for _, r in sub[sub.metric == metric].iterrows():
                mark = "*" if r["significant_p05"] else " "
                mark += "" if r["holm_significant"] else ("<-" if r["significant_p05"] else "")
                print(f"    {r['gene_dropped']}: diff={r['mean_diff']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                      f"d_z={r['cohens_dz']:+7.2f}  p={r['p_value']:.4f} {mark}")
        print()

    n_sig = int(result.significant_p05.sum())
    n_holm = int(result.holm_significant.sum())
    print(f"p<{ALPHA}: {n_sig} of {len(result)}   surviving Holm: {n_holm}")
    print("'<-' marks a difference significant at p<0.05 that does NOT survive Holm.")
    print(f"\nSaved: {OUT_CSV}")
