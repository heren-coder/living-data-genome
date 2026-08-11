"""Section 3.6: null distribution of the context-mismatch divergence, the
operating characteristic of the screen, and the basis for declaring xi.
Outputs: data/xi_operating_characteristic.csv, data/xi_null_summary.csv
"""
import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu
from generator import _simplex

D = "../data/"
EPS, SEEDS = 0.20, range(20)          # corruption magnitude, seeds
GRID = np.linspace(0.001, 0.20, 400)
REPORT = [0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150]

df = pd.read_csv(D + "scenarios.csv"); pil = pd.read_csv(D + "pi_lookup.csv")
jsd = lambda p, q: float(jensenshannon(p, q) ** 2)
rows, summ = [], []
for dom in ["RLV", "Healthcare"]:
    P = {r["context_label"]: np.array([r["pi_S"], r["pi_A"], r["pi_D"], r["pi_E"]])
         for _, r in pil[pil.domain == dom].iterrows()}
    labels = sorted(P)
    d = df[(df.domain == dom) & (df.stage_t == 2)].reset_index(drop=True)
    rho = np.vstack([_simplex(r) for r in d[["g_S", "g_A", "g_D", "g_E"]].values])
    feas = d["A_g"].values.astype(bool)
    pos, neg = [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        mis = rng.random(len(d)) < EPS
        decl = [rng.choice([c for c in labels if c != t]) if m else t
                for t, m in zip(d.context_label, mis)]
        X = np.array([jsd(rho[i], P[decl[i]]) for i in range(len(d))])
        pos.append(X[feas & mis]); neg.append(X[feas & ~mis])
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    auc = mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg))
    det = np.array([(pos > t).mean() for t in GRID])
    fa = np.array([(neg > t).mean() for t in GRID])
    j = (det - fa).argmax()
    summ.append(dict(domain=dom, n_corrupted=len(pos), n_correct=len(neg),
                     null_median=float(np.median(neg)), corrupted_median=float(np.median(pos)),
                     auc=float(auc), youden_xi=float(GRID[j]),
                     youden_detection=float(det[j]), youden_false_flag=float(fa[j]),
                     null_p95=float(np.percentile(neg, 95))))
    for x in REPORT:
        rows.append(dict(domain=dom, xi=x, detection=float((pos > x).mean()),
                         false_flag=float((neg > x).mean())))
pd.DataFrame(rows).to_csv(D + "xi_operating_characteristic.csv", index=False)
pd.DataFrame(summ).to_csv(D + "xi_null_summary.csv", index=False)
print("Saved: data/xi_operating_characteristic.csv, data/xi_null_summary.csv")
print(pd.DataFrame(summ)[["domain","null_median","auc","youden_xi"]].to_string(index=False))
