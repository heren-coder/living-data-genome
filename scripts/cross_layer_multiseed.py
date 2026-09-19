"""
cross_layer_multiseed.py -- the cross-layer coherence proxy over 50 generator
draws.

Why this script exists
----------------------
Every other stochastic quantity of the revision is reported over fifty
generator draws. The cross-layer proxy was not: cross_layer_coherence.csv
carries one row per domain, computed on the single submitted file, so the
section declares two thresholds and reports two numbers with no sampling
statement attached to either. This script closes that gap by calling
compute_cross_layer() unchanged, once per generator draw.

Nothing in the proxy is modified. Inputs are the per-draw files written by
generate_multiseed.py.

Outputs (data/multiseed/):
    cross_layer_multiseed.csv          one row per (domain, draw)
    cross_layer_multiseed_summary.csv  mean, sd and percentile CI per domain
"""
import os, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG   # noqa: E402
from rel_computation import compute_cross_layer                  # noqa: E402

M = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CONFIG = [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]
N = 50
TAU_LEV, TAU_BAL = 0.70, 0.80

rows = []
for k in range(N):
    sc = pd.read_csv(f"{M}scenarios_seed{k:02d}.csv")
    tr = pd.read_csv(f"{M}raw_traces_seed{k:02d}.csv")
    for dom, cfg in CONFIG:
        cl = compute_cross_layer(sc[sc.domain == dom], tr[tr.domain == dom], cfg,
                                 seed=RNG_SEED + 7)
        cl.update(domain=dom, k=k,
                  Coh=int(cl["CL_lev"] >= TAU_LEV and cl["CL_bal"] >= TAU_BAL))
        rows.append(cl)

d = pd.DataFrame(rows)
d.to_csv(M + "cross_layer_multiseed.csv", index=False)

out = []
for dom in ["RLV", "Healthcare"]:
    s = d[d.domain == dom]
    r = {"domain": dom, "n_draws": len(s)}
    for c in ["CL_lev", "CL_bal", "CKA_dna_rna", "CKA_rna_case", "CKA_dna_case"]:
        r[c + "_mean"] = s[c].mean(); r[c + "_sd"] = s[c].std(ddof=1)
        r[c + "_ci_lo"], r[c + "_ci_hi"] = np.percentile(s[c], [2.5, 97.5])
    r["Coh_rate"] = s.Coh.mean()
    r["CL_lev_min"], r["CL_bal_min"] = s.CL_lev.min(), s.CL_bal.min()
    out.append(r)
pd.DataFrame(out).to_csv(M + "cross_layer_multiseed_summary.csv", index=False)

print(f"{'domain':<12}{'CL_lev mean [95% CI]':>30}{'CL_bal mean [95% CI]':>30}{'Coh':>7}")
print("-" * 80)
for r in out:
    print(f"{r['domain']:<12}"
          f"{r['CL_lev_mean']:.3f} [{r['CL_lev_ci_lo']:.3f}, {r['CL_lev_ci_hi']:.3f}]".rjust(30)
          + f"{r['CL_bal_mean']:.3f} [{r['CL_bal_ci_lo']:.3f}, {r['CL_bal_ci_hi']:.3f}]".rjust(30)
          + f"{r['Coh_rate']:.2f}".rjust(7))
print(f"\nsingle file (k=0): "
      f"RLV {d[(d.domain=='RLV')&(d.k==0)].CL_lev.iloc[0]:.3f}/{d[(d.domain=='RLV')&(d.k==0)].CL_bal.iloc[0]:.3f}  "
      f"HC {d[(d.domain=='Healthcare')&(d.k==0)].CL_lev.iloc[0]:.3f}/{d[(d.domain=='Healthcare')&(d.k==0)].CL_bal.iloc[0]:.3f}")
print("Saved: data/multiseed/cross_layer_multiseed{,_summary}.csv")
