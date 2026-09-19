"""
cross_layer_arms.py -- does the cross-layer proxy respond to the validity arms?

Why this script exists
----------------------
Section 3.10 tests the aggregate against four convergent arms, which must lower
it, and three discriminant arms, which must not move it. The cross-layer proxy
of Section 2.8 was never put through that test: validity_arms.csv carries
A, TS, SR, GC and Rel, and no cross-layer column. The section therefore declares
two thresholds without evidence that the quantity behind them registers anything.

This script applies the same arm functions, unchanged, and computes the proxy on
each perturbed dataset. The arms perturb the gene coordinates, which are the
released-package view of the proxy; the protected trace and the case view are
left as they are, so the question the script answers is whether damaging the
package alone moves the three-way agreement.

Outputs: data/multiseed/cross_layer_arms.csv
"""
import os, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from generator import RNG_SEED                                   # noqa: E402
from rel_computation import compute_cross_layer                  # noqa: E402
import validity_arms as V                                        # noqa: E402  (re-runs its own outputs, deterministic)

D = os.path.join(_HERE, "..", "data") + os.sep
M = D + "multiseed" + os.sep
SEEDS = V.SEEDS
sc = pd.read_csv(D + "scenarios.csv")
tr = pd.read_csv(D + "raw_traces.csv")

ARMS = [("misalignment", V.c1_misalign, "convergent", 7),
        ("instability",  V.c2_unstable, "convergent", 7),
        ("releasability", V.c3_unreleasable, "convergent", 7),
        ("candidate order", V.d1_permute, "discriminant", 11),
        ("quality scale", V.d2_quality_scale, "discriminant", 11),
        ("enforced admissibility", V.x1_enforce, "inflation", 17)]

rows = []
for dom, cfg in V.CFG.items():
    # arm functions index positionally, so the per-domain subset needs a fresh index
    base_df = sc[sc.domain == dom].reset_index(drop=True)
    base_tr = tr[tr.domain == dom].reset_index(drop=True)
    b = compute_cross_layer(base_df, base_tr, cfg, seed=RNG_SEED + 7)
    rows.append(dict(domain=dom, arm="baseline", kind="-", seed=-1, **b))
    for seed in SEEDS:
        for name, fn, kind, off in ARMS:
            d = fn(base_df, cfg, np.random.default_rng(RNG_SEED + 101 * seed + off))
            cl = compute_cross_layer(d, base_tr, cfg, seed=RNG_SEED + 7)
            rows.append(dict(domain=dom, arm=name, kind=kind, seed=seed, **cl))

d = pd.DataFrame(rows)
agg = (d[d.seed >= 0].groupby(["domain", "arm", "kind"])
       .agg(CL_lev=("CL_lev", "mean"), CL_lev_sd=("CL_lev", "std"),
            CL_bal=("CL_bal", "mean"), CL_bal_sd=("CL_bal", "std")).reset_index())
base = d[d.seed < 0].set_index("domain")
agg["dCL_lev"] = [r.CL_lev - base.loc[r.domain, "CL_lev"] for r in agg.itertuples()]
agg["dCL_bal"] = [r.CL_bal - base.loc[r.domain, "CL_bal"] for r in agg.itertuples()]
agg.to_csv(M + "cross_layer_arms.csv", index=False)

va = pd.read_csv(D + "validity_arms.csv").set_index(["domain", "arm"])
print(f"{'domain':<11}{'arm':<24}{'kind':<14}{'dCL_lev':>9}{'dCL_bal':>9}{'dRel':>9}")
print("-" * 78)
for dom in ["RLV", "Healthcare"]:
    for r in agg[agg.domain == dom].itertuples():
        dr = va.loc[(dom, r.arm), "dRel"] if (dom, r.arm) in va.index else float("nan")
        print(f"{dom:<11}{r.arm:<24}{r.kind:<14}{r.dCL_lev:>9.4f}{r.dCL_bal:>9.4f}{dr:>9.4f}")
print("\nSaved: data/multiseed/cross_layer_arms.csv")
