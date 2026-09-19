"""
governance_federated_multiseed.py -- Table 3.4 / Figure 3.6 (governance) and
Figure 3.7 (federated agreement) over generator-level resampling.

Why this script exists
----------------------
governance_multiseed.py and federated_noise_sweep.py evaluate five evaluation
seeds on the single submitted scenarios.csv. This script calls their helpers
unchanged (one_arm, federated_agreement_demo, rel_with) on each of the 50
generator-resampled datasets, with a matched design: one evaluation seed per
dataset, equal to the dataset's own generator seed.

Rare-event handling: revoke, regeneration and
random-redraw counts are also reported as pooled COUNTS with a Clopper-Pearson
95 % interval on the pooled proportion, in addition to the seed-level bootstrap
CI on the rate. Pooled counts are the right summary when the per-seed
denominator (number of revoked artifacts) is small.

The A_triad and TS entering Rel are the per-dataset values from
data/multiseed/rel_multiseed.csv, so all Rel arms of a given seed share one
alignment/stability draw, exactly as in the submitted Table 3.4.

Inputs : data/multiseed/scenarios_seedNN.csv, multiseed_manifest.csv, rel_multiseed.csv
Outputs: data/multiseed/governance_multiseed50.csv          (per seed x domain)
         data/multiseed/governance_multiseed50_summary.csv  (means, CIs, pooled counts)
         data/multiseed/sweep_tightening50.csv              (Fig 3.6, per tightening)
         data/multiseed/sweep_node_noise50.csv              (Fig 3.7, per noise level)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG          # noqa: E402
from governance_multiseed import one_arm, rel_with, TIGHTENING, TIGHTEN_GRID  # noqa: E402
from federated_noise_sweep import RELATIVE_GRID, TOLERANCE              # noqa: E402
from day5_federated_crypto import federated_agreement_demo              # noqa: E402

D = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2025)


def boot_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = BRNG.integers(0, len(x), size=(B_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def clopper_pearson(k, n, level=0.95):
    if n == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


def stat_block(prefix, x):
    lo, hi = boot_ci(x)
    return {f"{prefix}_mean": float(np.nanmean(x)), f"{prefix}_sd": float(np.nanstd(x, ddof=1)),
            f"{prefix}_ci_lo": lo, f"{prefix}_ci_hi": hi}


if __name__ == "__main__":
    t0 = time.time()
    man = pd.read_csv(D + "multiseed_manifest.csv")
    relms = pd.read_csv(D + "rel_multiseed.csv").set_index(["k", "domain"])
    data = {}
    for _, m in man.iterrows():
        data[int(m.k)] = pd.read_csv(D + f"scenarios_seed{int(m.k):02d}.csv")
    seeds_of = {int(m.k): {"RLV": int(m.seed_rlv), "Healthcare": int(m.seed_hc)} for _, m in man.iterrows()}

    # ------------------------------------------------------------------ Table 3.4
    gov_rows = []
    for k, df in data.items():
        for domain, config in CONFIG:
            sub = df[df.domain == domain]
            s = seeds_of[k][domain]
            a, ts = relms.loc[(k, domain), "A_triad"], relms.loc[(k, domain), "TS_mean"]
            x = one_arm(sub, config, TIGHTENING, s)
            n_rev = int(round(x["revoke_rate"] * x["n_gen1"]))
            gov_rows.append({
                "k": k, "domain": domain, "eval_seed": s, "tightening": TIGHTENING,
                **x,
                "n_revoked": n_rev,
                "n_regen_ok": int(round(x["regeneration_success_rate"] * n_rev)) if n_rev else 0,
                "n_redraw_ok": int(round(x["random_redraw_success_rate"] * n_rev)) if n_rev else 0,
                "Rel_baseline": rel_with(a, ts, 1.0, 1.0),
                "Rel_discard": rel_with(a, ts, x["SR_discard"], x["GC_discard"]),
                "Rel_regenerate": rel_with(a, ts, x["SR_regenerate"], 1.0),
            })
    gov = pd.DataFrame(gov_rows)
    gov.to_csv(D + "governance_multiseed50.csv", index=False)

    summ = []
    for domain, g in gov.groupby("domain"):
        row = {"domain": domain, "n_seeds": int(g.k.nunique()), "tightening": TIGHTENING}
        for c in ["revoke_rate", "regeneration_success_rate", "random_redraw_success_rate",
                  "SR_discard", "GC_discard", "SR_regenerate",
                  "Rel_baseline", "Rel_discard", "Rel_regenerate"]:
            row.update(stat_block(c, g[c].values))
        gap = g["Rel_regenerate"].values - g["Rel_discard"].values
        row.update(stat_block("Rel_gap_regen_minus_discard", gap))
        # pooled counts + Clopper-Pearson (rare-event reporting)
        N, R = int(g.n_gen1.sum()), int(g.n_revoked.sum())
        RG, RD = int(g.n_regen_ok.sum()), int(g.n_redraw_ok.sum())
        row.update({"pooled_n_gen1": N, "pooled_n_revoked": R,
                    "pooled_revoke_rate": R / N, "pooled_revoke_cp_lo": clopper_pearson(R, N)[0],
                    "pooled_revoke_cp_hi": clopper_pearson(R, N)[1],
                    "pooled_n_regen_ok": RG, "pooled_regen_rate": RG / R if R else np.nan,
                    "pooled_regen_cp_lo": clopper_pearson(RG, R)[0], "pooled_regen_cp_hi": clopper_pearson(RG, R)[1],
                    "pooled_n_redraw_ok": RD, "pooled_redraw_rate": RD / R if R else np.nan,
                    "pooled_redraw_cp_lo": clopper_pearson(RD, R)[0], "pooled_redraw_cp_hi": clopper_pearson(RD, R)[1],
                    "min_revoked_per_seed": int(g.n_revoked.min())})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(D + "governance_multiseed50_summary.csv", index=False)
    print(f"Table 3.4 done ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ Figure 3.6
    sw = []
    for t in TIGHTEN_GRID:
        for domain, config in CONFIG:
            gaps, rev = [], []
            for k, df in data.items():
                sub = df[df.domain == domain]
                a, ts = relms.loc[(k, domain), "A_triad"], relms.loc[(k, domain), "TS_mean"]
                x = one_arm(sub, config, t, seeds_of[k][domain])
                gaps.append(rel_with(a, ts, x["SR_regenerate"], 1.0) - rel_with(a, ts, x["SR_discard"], x["GC_discard"]))
                rev.append(x["revoke_rate"])
            row = {"domain": domain, "tightening": t, "n_seeds": len(gaps)}
            row.update(stat_block("Rel_gap", np.array(gaps)))
            row.update(stat_block("revoke_rate", np.array(rev)))
            sw.append(row)
    pd.DataFrame(sw).to_csv(D + "sweep_tightening50.csv", index=False)
    print(f"Figure 3.6 sweep done ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ Figure 3.7
    fn = []
    for pct in RELATIVE_GRID:
        for domain, config in CONFIG:
            std = pct / 100.0 * TOLERANCE[domain]
            kap, una, adm = [], [], []
            for k, df in data.items():
                sub = df[df.domain == domain]
                r = federated_agreement_demo(domain, sub, config, n_nodes=4,
                                             node_noise_std=std, seed=seeds_of[k][domain])
                kap.append(r["fleiss_kappa"]); una.append(r["unanimous_rate"]); adm.append(r["mean_admissible_rate"])
            row = {"domain": domain, "relative_variance_pct": pct, "node_noise_std": std, "n_seeds": len(kap),
                   "regime": ("nominal" if 3 <= pct <= 9 else "stress_test" if pct >= 20 else "intermediate")}
            row.update(stat_block("fleiss_kappa", np.array(kap)))
            row.update(stat_block("unanimous_rate", np.array(una)))
            row["mean_admissible_rate"] = float(np.mean(adm))
            fn.append(row)
    fn = pd.DataFrame(fn)
    fn.to_csv(D + "sweep_node_noise50.csv", index=False)
    print(f"Figure 3.7 sweep done ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ printout
    print("\n=== Table 3.4 over 50 generator seeds (mean [95% CI]) ===")
    for _, r in summ.iterrows():
        print(f"\n--- {r.domain} ---   pooled: {r.pooled_n_revoked} revoked of {r.pooled_n_gen1}; "
              f"min revoked per seed = {r.min_revoked_per_seed}")
        for lab, c in [("Revoke rate", "revoke_rate"), ("Random-redraw success", "random_redraw_success_rate"),
                       ("Governed regeneration", "regeneration_success_rate"), ("Rel: baseline", "Rel_baseline"),
                       ("Rel: discard", "Rel_discard"), ("Rel: regenerate", "Rel_regenerate"),
                       ("Rel gap regen-discard", "Rel_gap_regen_minus_discard")]:
            print(f"  {lab:<24} {r[c + '_mean']:.3f}  [{r[c + '_ci_lo']:.3f}, {r[c + '_ci_hi']:.3f}]  sd {r[c + '_sd']:.3f}")
        print(f"  pooled revoke CP        [{r.pooled_revoke_cp_lo:.3f}, {r.pooled_revoke_cp_hi:.3f}]"
              f"   pooled regen CP [{r.pooled_regen_cp_lo:.3f}, {r.pooled_regen_cp_hi:.3f}]"
              f"   pooled redraw CP [{r.pooled_redraw_cp_lo:.3f}, {r.pooled_redraw_cp_hi:.3f}]")
    print("\n=== Figure 3.7 kappa, 50 seeds ===")
    piv = fn.pivot(index="relative_variance_pct", columns="domain", values=["fleiss_kappa_mean", "fleiss_kappa_sd"])
    print(piv.round(3).to_string())
    print(f"\nSaved: {D}governance_multiseed50*.csv, sweep_tightening50.csv, sweep_node_noise50.csv")
