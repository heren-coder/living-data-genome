"""Geometry of the declared region behind Corollary C2, per context.

For each context the marginal acceptance probability of a gene coordinate is the
fraction of repair-stage candidates of that context whose coordinate lies inside
its declared band. The corollary predicts that the pool-referenced attribution
concentrates on the coordinate with the smallest such probability, and that the
concentration grows with the margin between the two smallest. The concentration
itself is read from shapley_by_context.csv, so run shapley_by_context.py first.

Columns: tightest (coordinate with the smallest acceptance probability; in an
exact tie the last of the tied coordinates in S, A, D, E order), p_min (that
probability), gap (margin between the two smallest), pool_conc (within-context
modal share of the pool-referenced attribution).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, "..", "data") + os.sep
import pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG

CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
ORDER = {"RLV": ["dense_day", "sparse_day", "dense_night", "sparse_night"],
         "Healthcare": ["high_load_daytime", "low_load_daytime", "high_load_nighttime", "low_load_nighttime"]}

sc = pd.read_csv(D + "scenarios.csv")
conc = pd.read_csv(D + "shapley_by_context.csv").set_index(["domain", "context"])
rows = []
for dom, cfg in CFG.items():
    d = sc[(sc.domain == dom) & (sc.stage_t == 2)]
    for ctx in ORDER[dom]:
        x = d[d.context_label == ctx]
        bd = cfg.feasibility_bounds(ctx)
        p = {g: float(((x["g_" + g] >= lo) & (x["g_" + g] <= hi)).mean()) for g, (lo, hi) in bd.items()}
        ranked = sorted(p, key=lambda g: (p[g], -list(bd).index(g)))
        rows.append(dict(domain=dom, context=ctx, tightest=ranked[0],
                         p_min=round(p[ranked[0]], 3),
                         gap=round(p[ranked[1]] - p[ranked[0]], 3),
                         pool_conc=conc.loc[(dom, ctx), "pool_ref_modal_share"]))
out = pd.DataFrame(rows)
out.to_csv(D + "shapley_geometry.csv", index=False)
print(out.to_string(index=False))
print("correlation of gap with concentration:", round(float(out.gap.corr(out.pool_conc)), 3))
