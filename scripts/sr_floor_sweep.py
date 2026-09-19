"""
sr_floor_sweep.py -- quantitative influence of the numerical floor on
admissibility survival.

Why this script exists
----------------------
compute_rel() evaluates Rel per event with SR_event clipped at eps = 1e-6
(rel_computation.py line ~251); rel_with() applies the same floor to SR and
GC. The exact weighted geometric mean (Eq. 2.17) sends Rel to 0 when SR = 0
(weakest-link property, Proposition P-WL). With the floor, a completely
failed candidate family instead contributes

    Rel_floor = eps^{w_SR} * (A_triad * TS * GC)^{w}   ~  0.032 * (...)^{0.25}

so the floor is a numerical convention that lifts Rel off zero by a bounded
amount, eps^{w_SR}. This script measures that amount (i) at the operating
tolerance, where SR = 0 is rare, and (ii) under progressively tightened
tolerance delta, where candidate families approach complete failure and the
floor becomes material. It uses the 50 generator-resampled datasets so every
number carries a CI.

Two Rel definitions are reported side by side:
    Rel_exact  : eps = 0            (weakest-link, per Eq. 2.17)
    Rel_floor  : eps in EPS_GRID    (eps = 1e-6 is the submitted convention)
and the deviation Rel_floor - Rel_exact, both event-averaged and restricted
to the SR = 0 events.

A_triad and TS are held at their per-dataset measured values (as in the
delta sweep of sensitivity_sweeps.py, since delta enters only through A_g).

Inputs : data/multiseed/scenarios_seedNN.csv, rel_multiseed.csv
Outputs: data/multiseed/sr_floor_eps_sweep.csv       (per eps x domain, operating delta)
         data/multiseed/sr_floor_delta_sweep.csv     (per delta x domain, eps = 0 vs 1e-6)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG   # noqa: E402
from rel_computation import REL_WEIGHTS                            # noqa: E402
from sensitivity_sweeps import admissible_at, OPERATING_DELTA      # noqa: E402

D = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CONFIG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
EPS_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
DELTA_GRID = [0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06]
W = REL_WEIGHTS
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2026)


def boot_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[BRNG.integers(0, len(x), size=(B_BOOT, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def sr_per_event(sub, config, delta):
    rep = sub[sub.stage_t == 2]
    adm = admissible_at(rep, config, delta)
    return pd.Series(adm, index=rep.index).groupby(rep["event_id"].values).mean().values


def rel_events(a, ts, sr, eps):
    sr_eff = np.maximum(sr, eps) if eps > 0 else sr
    return (a ** W["A_triad"]) * (ts ** W["TS"]) * (sr_eff ** W["SR"]) * (1.0 ** W["GC"])


if __name__ == "__main__":
    man = pd.read_csv(D + "multiseed_manifest.csv")
    relms = pd.read_csv(D + "rel_multiseed.csv").set_index(["k", "domain"])
    data = {int(m.k): pd.read_csv(D + f"scenarios_seed{int(m.k):02d}.csv") for _, m in man.iterrows()}

    # ------------------------------------------------ (i) eps sweep at operating delta
    rows = []
    for dom, cfg in CONFIG.items():
        per_eps = {e: {"rel": [], "dev": [], "dev_sr0": [], "n_sr0": [], "frac_sr0": []} for e in EPS_GRID}
        for k, df in data.items():
            sub = df[df.domain == dom]
            a, ts = relms.loc[(k, dom), "A_triad"], relms.loc[(k, dom), "TS_mean"]
            sr = sr_per_event(sub, cfg, OPERATING_DELTA[dom])
            exact = rel_events(a, ts, sr, 0.0)
            z = sr == 0
            for e in EPS_GRID:
                r = rel_events(a, ts, sr, e)
                per_eps[e]["rel"].append(r.mean())
                per_eps[e]["dev"].append((r - exact).mean())
                per_eps[e]["dev_sr0"].append((r[z] - exact[z]).mean() if z.any() else np.nan)
                per_eps[e]["n_sr0"].append(int(z.sum()))
                per_eps[e]["frac_sr0"].append(float(z.mean()))
        for e in EPS_GRID:
            v = per_eps[e]
            row = {"domain": dom, "delta": OPERATING_DELTA[dom], "eps": e,
                   "floor_factor_eps_pow_wSR": e ** W["SR"] if e > 0 else 0.0,
                   "n_seeds": len(v["rel"]),
                   "SR_zero_events_mean": float(np.mean(v["n_sr0"])),
                   "SR_zero_frac_mean": float(np.mean(v["frac_sr0"]))}
            for c in ["rel", "dev", "dev_sr0"]:
                lo, hi = boot_ci(v[c])
                row[f"{c}_mean"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = float(np.nanmean(v[c])), lo, hi
            rows.append(row)
    eps_df = pd.DataFrame(rows)
    eps_df.to_csv(D + "sr_floor_eps_sweep.csv", index=False)

    # ------------------------------------------------ (ii) delta sweep, eps = 0 vs 1e-6
    rows = []
    for dom, cfg in CONFIG.items():
        for delta in DELTA_GRID:
            acc = {"Rel_exact": [], "Rel_floor": [], "dev": [], "frac_sr0": [], "SR_mean": []}
            for k, df in data.items():
                sub = df[df.domain == dom]
                a, ts = relms.loc[(k, dom), "A_triad"], relms.loc[(k, dom), "TS_mean"]
                sr = sr_per_event(sub, cfg, delta)
                ex, fl = rel_events(a, ts, sr, 0.0), rel_events(a, ts, sr, 1e-6)
                acc["Rel_exact"].append(ex.mean()); acc["Rel_floor"].append(fl.mean())
                acc["dev"].append((fl - ex).mean()); acc["frac_sr0"].append(float((sr == 0).mean()))
                acc["SR_mean"].append(sr.mean())
            row = {"domain": dom, "delta": delta, "n_seeds": len(acc["dev"])}
            for c, v in acc.items():
                lo, hi = boot_ci(v)
                row[f"{c}_mean"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = float(np.mean(v)), lo, hi
            # bound: mean deviation <= frac_sr0 * eps^wSR * (a*ts)^{...}; report the ratio to the bound
            rows.append(row)
    dl = pd.DataFrame(rows)
    dl.to_csv(D + "sr_floor_delta_sweep.csv", index=False)

    print("=== (i) eps sweep at operating delta (50 seeds) ===")
    print(eps_df[["domain", "eps", "floor_factor_eps_pow_wSR", "SR_zero_frac_mean", "rel_mean",
                  "dev_mean", "dev_ci_hi", "dev_sr0_mean"]].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("\n=== (ii) delta sweep, exact vs floored Rel (eps = 1e-6), 50 seeds ===")
    print(dl[["domain", "delta", "SR_mean_mean", "frac_sr0_mean", "Rel_exact_mean", "Rel_floor_mean",
              "dev_mean", "dev_ci_lo", "dev_ci_hi"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {D}sr_floor_eps_sweep.csv, sr_floor_delta_sweep.csv")
