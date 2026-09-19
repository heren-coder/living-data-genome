"""
rel_multiseed.py -- Rel, its components, the supervised baseline and the
gene ablations over generator-level resampling.

Why this script exists
----------------------
rel_seed_robustness.py and day5_ablation.py vary only the evaluation seed
(case-facing noise, train/test split) on one fixed scenarios.csv, so their
"five-seed" spread reflects A_triad and the baseline alone. This script
re-runs the SAME estimators (compute_rel, run_ablation) on each of the
generator-resampled datasets written by generate_multiseed.py, so that TS,
SR and the ablation deltas acquire sampling uncertainty as well.

Design
------
* One evaluation seed per generator seed (matched design): evaluation seed
  = generator seed of that dataset. Uncertainty is therefore total
  (data + evaluation); the
  evaluation-only spread of the submitted paper is retained as a separate
  column for comparison.
* Ablation deltas are paired within seed (same dataset for full and ablated
  arms), then summarised across seeds -- the same pairing logic as
  ablation_significance.py, now over 50 seeds instead of 5.
* Interval: percentile bootstrap, B = 10000, 95 %.  GC is the placeholder 1.0
  as in the main results; the measured-GC arm is handled by
  governance_multiseed.py.

Inputs : data/multiseed/scenarios_seedNN.csv, data/multiseed/multiseed_manifest.csv
Outputs: data/multiseed/rel_multiseed.csv           (one row per seed x domain)
         data/multiseed/rel_multiseed_summary.csv   (mean, SD, 95 % CI)
         data/multiseed/ablation_multiseed.csv      (one row per seed x domain x gene)
         data/multiseed/ablation_multiseed_summary.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG  # noqa: E402
from rel_computation import compute_rel, REL_WEIGHTS         # noqa: E402
from day5_ablation import run_ablation                        # noqa: E402

D = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2024)


def boot_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = BRNG.integers(0, len(x), size=(B_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(m, a)), float(np.quantile(m, 1 - a)))


def summarise(df, keys, cols):
    rows = []
    for kv, g in df.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        row = dict(zip(keys, kv))
        row["n_seeds"] = int(g["k"].nunique())
        for c in cols:
            x = g[c].values.astype(float)
            lo, hi = boot_ci(x)
            row[f"{c}_mean"] = float(np.nanmean(x))
            row[f"{c}_sd"] = float(np.nanstd(x, ddof=1))
            row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    man = pd.read_csv(D + "multiseed_manifest.csv")
    rel_rows, abl_rows = [], []
    for _, m in man.iterrows():
        k = int(m.k)
        df = pd.read_csv(D + f"scenarios_seed{k:02d}.csv")
        for dom in ["RLV", "Healthcare"]:
            sub = df[df.domain == dom]
            eval_seed = int(m.seed_rlv if dom == "RLV" else m.seed_hc)
            summ, per_event, base = compute_rel(dom, sub, CFG[dom], seed=eval_seed)
            full_rel = float(summ["Rel_mean"])
            rel_rows.append(dict(
                k=k, domain=dom, eval_seed=eval_seed,
                A_triad=summ["A_triad"],
                CKA_genesis_repair=summ["CKA_genesis_repair"],
                CKA_genesis_context=summ["CKA_genesis_context"],
                CKA_repair_context=summ["CKA_repair_context"],
                TS_mean=summ["TS_mean"], SR_mean=summ["SR_mean"],
                SR_zero_events=int((per_event["SR_event"] == 0).sum()),
                Rel_mean=full_rel, Rel_event_sd=summ["Rel_std"],
                baseline_balanced_accuracy=base.get("balanced_accuracy", np.nan),
                baseline_f1=base.get("f1", np.nan),
            ))
            abl = run_ablation(dom, sub, CFG[dom], full_rel=full_rel,
                               seed=eval_seed, n_seeds=1)
            abl.insert(0, "k", k)
            abl_rows.append(abl)
        if k % 10 == 9:
            print(f"  seed {k + 1}/{len(man)} done ({time.time() - t0:.0f}s)")

    rel = pd.DataFrame(rel_rows)
    rel.to_csv(D + "rel_multiseed.csv", index=False)
    abl = pd.concat(abl_rows, ignore_index=True)
    abl.to_csv(D + "ablation_multiseed.csv", index=False)

    rel_cols = ["A_triad", "TS_mean", "SR_mean", "Rel_mean", "SR_zero_events",
                "baseline_balanced_accuracy", "baseline_f1"]
    rel_summ = summarise(rel, ["domain"], rel_cols)
    rel_summ.to_csv(D + "rel_multiseed_summary.csv", index=False)

    abl_cols = ["Rel", "Rel_delta_vs_full", "A_triad_mean", "TS_mean", "SR",
                "baseline_balanced_accuracy_mean"]
    abl_summ = summarise(abl, ["domain", "gene_dropped"], abl_cols)
    # paired sign test across seeds: fraction of seeds with delta < 0
    frac = abl.groupby(["domain", "gene_dropped"])["Rel_delta_vs_full"].apply(lambda x: float((x < 0).mean()))
    abl_summ = abl_summ.merge(frac.rename("frac_seeds_delta_negative").reset_index(), on=["domain", "gene_dropped"])
    abl_summ.to_csv(D + "ablation_multiseed_summary.csv", index=False)

    print(f"\n=== rel_multiseed: {len(man)} generator seeds, {time.time() - t0:.0f}s ===")
    show = ["domain", "n_seeds"] + [f"{c}_{s}" for c in ["A_triad", "TS_mean", "SR_mean", "Rel_mean"] for s in ["mean", "sd", "ci_lo", "ci_hi"]]
    print(rel_summ[show].round(4).T.to_string())
    print("\n--- ablation: Rel delta vs full, paired within seed ---")
    print(abl_summ[["domain", "gene_dropped", "Rel_delta_vs_full_mean", "Rel_delta_vs_full_ci_lo",
                    "Rel_delta_vs_full_ci_hi", "frac_seeds_delta_negative"]].round(4).to_string(index=False))
    print(f"\nSaved: {D}rel_multiseed*.csv, ablation_multiseed*.csv")
