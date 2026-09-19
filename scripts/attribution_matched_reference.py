"""
attribution_matched_reference.py -- controlled comparison of the three
attribution methods under matched reference distributions.

Why this script exists
----------------------
lime_probe.py reports that the counterfactual binding coordinate and the
LIME surrogate agree on ~85-89 % of released candidates while exact
interventional Shapley agrees with either at chance (~25 %). The submitted
text attributes this to the reference distribution: LIME perturbs LOCALLY
around the candidate (Gaussian, sigma = 0.5 x bound half-width) whereas
Shapley draws its background from the context POOL. This script tests that
reading with matched references rather than asserting it.

Design: 2 methods x 2 references, plus a kernel-width sweep.
    Shapley | pool reference   (submitted)
    Shapley | local reference  (background = same local Gaussian as LIME)
    LIME    | local reference  (submitted)
    LIME    | pool reference   (perturbations drawn from the context pool)
The counterfactual binding coordinate has no reference distribution (it is
the coordinate with the smallest slack to the admissible bound) and is the
common comparator. If agreement tracks the REFERENCE and not the METHOD,
the submitted interpretation holds. The width sweep varies sigma_local as a
multiple of the bound half-width (0.25, 0.5, 1, 2, 4); as the local
reference widens it should approach the pool result.

All estimators are copied unchanged from lime_probe.py; only the sampling
of the background / perturbation set is parameterised. Evaluated on the
first N_SEEDS generator-resampled datasets, released candidates only
(A_imm: admissible, quality >= 0.50, TS >= 0.90, as submitted).

Outputs: data/multiseed/attribution_matched.csv          (per seed x domain x cell)
         data/multiseed/attribution_matched_summary.csv  (mean, 95 % CI)
         data/multiseed/attribution_width_sweep.csv      (per width)
"""
import os
import sys
import time
import itertools
from math import factorial

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG  # noqa: E402

D = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
QMIN, TAU, STS = 0.50, 0.90, 8.0
NBG, NLIME = 200, 500
N_SEEDS = int(os.environ.get("LDG_N_SEEDS", 50))
SEED_START = int(os.environ.get("LDG_SEED_START", 0))
SUMMARY_ONLY = os.environ.get("LDG_SUMMARY_ONLY", "0") == "1"
N_CAND = 150          # released candidates sampled per dataset x domain (exact Shapley is O(2^4 * NBG) each)
WIDTHS = [0.25, 0.5, 1.0, 2.0, 4.0]
SUBS = [s for r in range(5) for s in itertools.combinations(range(4), r)]
SHAP_W = {s: factorial(len(s)) * factorial(3 - len(s)) / factorial(4) for s in SUBS if len(s) <= 3}
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2029)


def boot_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[BRNG.integers(0, len(x), size=(B_BOOT, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load(df, dom):
    d = df[df.domain == dom]
    a = d[d.stage_t == 1].sort_values(["event_id", "candidate_id"]).reset_index(drop=True)
    b = d[d.stage_t == 2].sort_values(["event_id", "candidate_id"]).reset_index(drop=True)
    g1 = a[["g_S", "g_A", "g_D", "g_E"]].values; g = b[["g_S", "g_A", "g_D", "g_E"]].values
    q = b[["q_S", "q_A", "q_D", "q_E"]].values; qn = q / q.sum(axis=1, keepdims=True)
    ts = np.exp(-((g - g1) * (STS * qn) * (g - g1)).sum(axis=1))
    return b, g, q, ts


def bounds(b, cfg):
    return {c: cfg.feasibility_bounds(c) for c in b.context_label.unique()}


def halfwidth(bd, ctx):
    return np.array([(bd[ctx][gn][1] - bd[ctx][gn][0]) / 2 for gn in bd[ctx]])


def slacks(b, g, bd):
    sl = np.zeros_like(g)
    for i, c in enumerate(b.context_label):
        for k, gn in enumerate(bd[c]):
            lo, hi = bd[c][gn]; sl[i, k] = min(g[i, k] - lo, hi - g[i, k])
    return sl


def inside(X, ctx, bd):
    ok = np.ones(len(X), bool)
    for k, gn in enumerate(bd[ctx]):
        lo, hi = bd[ctx][gn]; ok &= (X[:, k] >= lo) & (X[:, k] <= hi)
    return ok


def sample_ref(kind, g_i, ctx, bd, pools, n, rng, width=0.5):
    """Background / perturbation set for candidate i under a given reference."""
    if kind == "pool":
        return pools[ctx][rng.integers(0, len(pools[ctx]), n)]
    return g_i + rng.normal(0, width * halfwidth(bd, ctx), size=(n, 4))       # local


def shapley(g, b, bd, pools, idx, ref, rng, width=0.5):
    S = np.zeros((len(idx), 4))
    for n, i in enumerate(idx):
        ctx = b.context_label[i]
        smp = sample_ref(ref, g[i], ctx, bd, pools, NBG, rng, width)
        v = {}
        for s in SUBS:
            X = smp.copy()
            for k in s: X[:, k] = g[i, k]
            v[s] = inside(X, ctx, bd).mean()
        for k in range(4):
            S[n, k] = sum(SHAP_W[s] * (v[tuple(sorted(s + (k,)))] - v[s]) for s in SUBS if k not in s)
    return S


def lime(g, b, bd, pools, idx, ref, rng, width=0.5):
    L = np.zeros((len(idx), 4))
    for n, i in enumerate(idx):
        ctx = b.context_label[i]; hw = halfwidth(bd, ctx)
        Z = sample_ref(ref, g[i], ctx, bd, pools, NLIME, rng, width)
        y = inside(Z, ctx, bd).astype(float)
        d = np.sqrt((((Z - g[i]) / hw) ** 2).sum(axis=1))
        w = np.exp(-(d ** 2) / (0.75 ** 2 * 4))
        Zs = (Z - g[i]) / hw
        A = np.hstack([np.ones((len(Z), 1)), Zs]); W = np.diag(w)
        beta = np.linalg.solve(A.T @ W @ A + 1e-6 * np.eye(5), A.T @ W @ y)
        L[n] = beta[1:]
    return L


def top(M):
    return np.abs(M).argmax(axis=1)


if __name__ == "__main__":
    t0 = time.time()
    man = pd.read_csv(D + "multiseed_manifest.csv")
    rows, wrows = [], []
    for k in ([] if SUMMARY_ONLY else range(SEED_START, SEED_START + N_SEEDS)):
        df = pd.read_csv(D + f"scenarios_seed{k:02d}.csv")
        for dom in CFG:
            b, g, q, ts = load(df, dom); bd = bounds(b, CFG[dom]); sl = slacks(b, g, bd)
            pools = {c: g[(b.context_label == c).values] for c in bd}
            Aimm = b.A_g.values.astype(bool) & (q >= QMIN).all(axis=1) & (ts >= TAU)
            rng = np.random.default_rng(RNG_SEED + 1000 * k + 17)
            idx = np.where(Aimm)[0]
            idx = rng.choice(idx, size=min(N_CAND, len(idx)), replace=False)
            bind = sl[idx].argmin(axis=1)
            cells = {
                ("shapley", "pool"):  top(shapley(g, b, bd, pools, idx, "pool", rng)),
                ("shapley", "local"): top(shapley(g, b, bd, pools, idx, "local", rng)),
                ("lime", "local"):    top(lime(g, b, bd, pools, idx, "local", rng)),
                ("lime", "pool"):     top(lime(g, b, bd, pools, idx, "pool", rng)),
            }
            for (m, r), t in cells.items():
                rows.append({"k": k, "domain": dom, "method": m, "reference": r, "n": len(idx),
                             "agree_with_counterfactual": float((t == bind).mean())})
            # cross-method agreement at matched reference
            for r in ["pool", "local"]:
                rows.append({"k": k, "domain": dom, "method": "shapley_vs_lime", "reference": r, "n": len(idx),
                             "agree_with_counterfactual": float((cells[("shapley", r)] == cells[("lime", r)]).mean())})
            for w in WIDTHS:
                ts_ = top(shapley(g, b, bd, pools, idx, "local", rng, width=w))
                tl_ = top(lime(g, b, bd, pools, idx, "local", rng, width=w))
                wrows.append({"k": k, "domain": dom, "width_x_halfwidth": w,
                              "shapley_local_vs_cf": float((ts_ == bind).mean()),
                              "lime_local_vs_cf": float((tl_ == bind).mean()),
                              "shapley_vs_lime": float((ts_ == tl_).mean())})
        print(f"  seed {k} done ({time.time() - t0:.0f}s)")

    for name, new in [("attribution_matched.csv", rows), ("attribution_width_sweep.csv", wrows)]:
        p = D + name
        if new:
            new = pd.DataFrame(new)
            if SEED_START > 0 and os.path.exists(p):
                old = pd.read_csv(p); old = old[~old.k.isin(new.k.unique())]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(p, index=False)
    res = pd.read_csv(D + "attribution_matched.csv"); wid = pd.read_csv(D + "attribution_width_sweep.csv")

    out = []
    for (dom, m, r), g_ in res.groupby(["domain", "method", "reference"]):
        lo, hi = boot_ci(g_.agree_with_counterfactual)
        out.append({"domain": dom, "method": m, "reference": r, "n_seeds": int(g_.k.nunique()),
                    "agreement_mean": float(g_.agree_with_counterfactual.mean()), "ci_lo": lo, "ci_hi": hi, "chance": 0.25})
    out = pd.DataFrame(out); out.to_csv(D + "attribution_matched_summary.csv", index=False)
    ws = wid.groupby(["domain", "width_x_halfwidth"])[["shapley_local_vs_cf", "lime_local_vs_cf", "shapley_vs_lime"]].mean().reset_index()

    print(f"\n=== 2x2 matched-reference agreement with the counterfactual binding coordinate ({res.k.nunique()} seeds) ===")
    for dom in CFG:
        print(f"\n--- {dom} ---")
        o = out[out.domain == dom]
        for _, r in o.iterrows():
            print(f"  {r.method:<16} {r.reference:<6} {r.agreement_mean:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
    print("\n=== local-reference width sweep (x bound half-width) ===")
    print(ws.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {D}attribution_matched*.csv, attribution_width_sweep.csv")
