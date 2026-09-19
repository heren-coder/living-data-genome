"""
xi_calibration_split.py -- declaring the context-mismatch tolerance xi on
calibration data and evaluating it on held-out data.

Why this script exists
----------------------
xi_operating_characteristic.py derives the Youden-optimal tolerance
(argmax detection - false_flag) on the same scenarios.csv on which Table 3.6
and the routing results are then evaluated; the declared xi = 0.030 sits at
the mean of the two domains' Youden points (0.0319, 0.0269). That choice is
outcome-informed. This script separates the two roles:

  calibration set : generator seeds k = 0..N_CAL-1   (N_CAL = 10)
  evaluation set  : generator seeds k = N_CAL..49    (40 datasets)

Three tolerance rules are declared on the calibration set only:
  xi_youden : mean Youden point across calibration datasets (per domain)
  xi_fa5    : 95th percentile of the null divergence (false-flag <= 5 %,
              detection-blind -- a rule an institution could set a priori)
  xi_decl   : 0.030, the submitted value, kept for continuity
and all three are then evaluated on the held-out set: detection, false flag,
and the four-route outcome of Eq. (2.20) at eta = 0.010. A sensitivity band
over xi in [0.010, 0.080] on the evaluation set closes the analysis.

Corruption model is unchanged from the submitted scripts: a fraction EPS of
released candidates has its declared context replaced by a random other
context; N_CORR corruption draws per dataset.

Inputs : data/multiseed/scenarios_seedNN.csv, data/pi_lookup.csv
Outputs: data/multiseed/xi_calibration.csv     (per calibration dataset)
         data/multiseed/xi_rules.csv           (the three declared rules)
         data/multiseed/xi_evaluation.csv      (held-out performance per rule)
         data/multiseed/xi_evaluation_band.csv (held-out sweep over xi)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import RNG_SEED, _simplex  # noqa: E402

D = os.path.join(_HERE, "..", "data") + os.sep
M = D + "multiseed" + os.sep
N_CAL, EPS, N_CORR, ETA, XI_DECL = 10, 0.20, 5, 0.010, 0.030
GRID = np.linspace(0.001, 0.20, 400)
BAND = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.065, 0.080]
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2027)


def boot_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[BRNG.integers(0, len(x), size=(B_BOOT, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def jsd(p, q):
    return float(jensenshannon(p, q) ** 2)


def prepare(df, pil, dom):
    P = {r.context_label: np.array([r.pi_S, r.pi_A, r.pi_D, r.pi_E]) for _, r in pil[pil.domain == dom].iterrows()}
    d = df[(df.domain == dom) & (df.stage_t == 2)].reset_index(drop=True)
    rho = np.vstack([_simplex(r) for r in d[["g_S", "g_A", "g_D", "g_E"]].values])
    Ag = d.A_g.values.astype(bool)
    labels = sorted(P)
    Xi0 = np.array([jsd(rho[i], P[d.context_label[i]]) for i in range(len(d))])
    best = np.array([min(jsd(rho[i], P[c]) for c in labels) for i in range(len(d))])
    return d, rho, Ag, P, labels, Xi0, best


def corrupted_divergences(d, rho, Ag, P, labels, seed_base):
    pos, neg = [], []
    for s in range(N_CORR):
        rng = np.random.default_rng(seed_base + s)
        mis = rng.random(len(d)) < EPS
        decl = [rng.choice([c for c in labels if c != t]) if m else t for t, m in zip(d.context_label, mis)]
        X = np.array([jsd(rho[i], P[decl[i]]) for i in range(len(d))])
        pos.append(X[Ag & mis]); neg.append(X[Ag & ~mis])
    return np.concatenate(pos), np.concatenate(neg)


def routes(Ag, Xi0, best, xi):
    Actx = (Xi0 - best) >= ETA
    return {"proceed": float((Ag & (Xi0 <= xi)).mean()),
            "context_review": float((Ag & (Xi0 > xi) & Actx).mean()),
            "quarantine_repair": float((Ag & (Xi0 > xi) & ~Actx).mean())}


if __name__ == "__main__":
    pil = pd.read_csv(D + "pi_lookup.csv")
    man = pd.read_csv(M + "multiseed_manifest.csv")
    ks = sorted(int(k) for k in man.k)
    cal_ks, eval_ks = ks[:N_CAL], ks[N_CAL:]

    # ------------------------------------------------------------- calibration
    cal_rows = []
    for k in cal_ks:
        df = pd.read_csv(M + f"scenarios_seed{k:02d}.csv")
        for dom in ["RLV", "Healthcare"]:
            d, rho, Ag, P, labels, Xi0, best = prepare(df, pil, dom)
            pos, neg = corrupted_divergences(d, rho, Ag, P, labels, seed_base=1000 * k)
            det = np.array([(pos > t).mean() for t in GRID]); fa = np.array([(neg > t).mean() for t in GRID])
            j = int((det - fa).argmax())
            cal_rows.append({"k": k, "domain": dom, "role": "calibration",
                             "youden_xi": float(GRID[j]), "youden_det": float(det[j]), "youden_fa": float(fa[j]),
                             "null_p95": float(np.percentile(neg, 95)), "null_median": float(np.median(neg))})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(M + "xi_calibration.csv", index=False)

    rules = []
    for dom, g in cal.groupby("domain"):
        yl, yh = boot_ci(g.youden_xi); pl, ph = boot_ci(g.null_p95)
        rules += [{"domain": dom, "rule": "xi_youden", "xi": float(g.youden_xi.mean()), "ci_lo": yl, "ci_hi": yh,
                   "basis": "mean Youden point over calibration datasets"},
                  {"domain": dom, "rule": "xi_fa5", "xi": float(g.null_p95.mean()), "ci_lo": pl, "ci_hi": ph,
                   "basis": "95th percentile of null divergence (false-flag-controlled)"},
                  {"domain": dom, "rule": "xi_decl", "xi": XI_DECL, "ci_lo": np.nan, "ci_hi": np.nan,
                   "basis": "submitted value"}]
    rules = pd.DataFrame(rules)
    rules.to_csv(M + "xi_rules.csv", index=False)
    xi_of = {(r.domain, r.rule): r.xi for _, r in rules.iterrows()}

    # ------------------------------------------------------------- evaluation
    ev_rows, band_rows = [], []
    for k in eval_ks:
        df = pd.read_csv(M + f"scenarios_seed{k:02d}.csv")
        for dom in ["RLV", "Healthcare"]:
            d, rho, Ag, P, labels, Xi0, best = prepare(df, pil, dom)
            pos, neg = corrupted_divergences(d, rho, Ag, P, labels, seed_base=1000 * k)
            for rule in ["xi_youden", "xi_fa5", "xi_decl"]:
                xi = xi_of[(dom, rule)]
                ev_rows.append({"k": k, "domain": dom, "rule": rule, "xi": xi,
                                "detection": float((pos > xi).mean()), "false_flag": float((neg > xi).mean()),
                                **routes(Ag, Xi0, best, xi)})
            for xi in BAND:
                band_rows.append({"k": k, "domain": dom, "xi": xi,
                                  "detection": float((pos > xi).mean()), "false_flag": float((neg > xi).mean()),
                                  **routes(Ag, Xi0, best, xi)})
    ev = pd.DataFrame(ev_rows); band = pd.DataFrame(band_rows)

    def summ(df, keys):
        out = []
        for kv, g in df.groupby(keys):
            kv = kv if isinstance(kv, tuple) else (kv,)
            row = dict(zip(keys, kv)); row["n_eval_datasets"] = int(g.k.nunique()); row["xi"] = float(g.xi.iloc[0])
            for c in ["detection", "false_flag", "proceed", "context_review", "quarantine_repair"]:
                lo, hi = boot_ci(g[c]); row[f"{c}_mean"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = float(g[c].mean()), lo, hi
            out.append(row)
        return pd.DataFrame(out)

    evs = summ(ev, ["domain", "rule"]); evs.to_csv(M + "xi_evaluation.csv", index=False)
    bds = summ(band, ["domain", "xi"]); bds.to_csv(M + "xi_evaluation_band.csv", index=False)

    print("=== declared rules (calibration set, k = 0..%d) ===" % (N_CAL - 1))
    print(rules[["domain", "rule", "xi", "ci_lo", "ci_hi"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== held-out evaluation (k = %d..%d), mean [95%% CI] ===" % (eval_ks[0], eval_ks[-1]))
    for _, r in evs.iterrows():
        print(f"{r.domain:<11} {r.rule:<10} xi={r.xi:.4f}  det {r.detection_mean:.3f} [{r.detection_ci_lo:.3f},{r.detection_ci_hi:.3f}]"
              f"  fa {r.false_flag_mean:.3f} [{r.false_flag_ci_lo:.3f},{r.false_flag_ci_hi:.3f}]"
              f"  routes P/CR/Q {r.proceed_mean:.3f}/{r.context_review_mean:.3f}/{r.quarantine_repair_mean:.3f}")
    print("\n=== held-out sensitivity band ===")
    print(bds[["domain", "xi", "detection_mean", "false_flag_mean", "proceed_mean", "context_review_mean",
               "quarantine_repair_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {M}xi_calibration.csv, xi_rules.csv, xi_evaluation.csv, xi_evaluation_band.csv")
