"""Section 3.2.1: artifact gate A_R, immune condition A_imm, counterfactual
release probe, exact Shapley cross-check, and the explanation payload.
Outputs: data/release_gating.csv, data/counterfactual_probe.csv, data/expl_payload_stats.csv
"""
import itertools, json, hashlib
from math import factorial
import numpy as np, pandas as pd
from generator import RLV_CONFIG, HEALTHCARE_CONFIG

D = "../data/"
QMIN, TAU, STS = 0.50, 0.90, 8.0          # Table 3.9
CFG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
NAMES = {"RLV": ["Speed", "Attention", "Density", "Environment"],
         "Healthcare": ["AccessEscalationRate", "AuditCoverage",
                        "ConcurrentRecordDensity", "AccessControlRegime"]}
NBG, RNG = 200, np.random.default_rng(7)


def load(dom):
    df = pd.read_csv(D + "scenarios.csv")
    d = df[df.domain == dom]
    a = d[d.stage_t == 1].sort_values(["event_id", "candidate_id"]).reset_index(drop=True)
    b = d[d.stage_t == 2].sort_values(["event_id", "candidate_id"]).reset_index(drop=True)
    g1 = a[["g_S", "g_A", "g_D", "g_E"]].values
    g = b[["g_S", "g_A", "g_D", "g_E"]].values
    q = b[["q_S", "q_A", "q_D", "q_E"]].values
    qn = q / q.sum(axis=1, keepdims=True)
    ts = np.exp(-((g - g1) * (STS * qn) * (g - g1)).sum(axis=1))
    return b, g, q, ts


def slacks(b, g, cfg):
    bd = {c: cfg.feasibility_bounds(c) for c in b.context_label.unique()}
    sl = np.zeros_like(g)
    for i, c in enumerate(b.context_label):
        for k, gn in enumerate(bd[c]):
            lo, hi = bd[c][gn]
            sl[i, k] = min(g[i, k] - lo, hi - g[i, k])
    return sl, bd


def shapley(g, b, bd, idx, cfg):
    subs = [s for r in range(5) for s in itertools.combinations(range(4), r)]
    pools = {c: g[(b.context_label == c).values] for c in bd}
    S = np.zeros((len(idx), 4))
    for n, i in enumerate(idx):
        ctx = b.context_label[i]
        smp = pools[ctx][RNG.integers(0, len(pools[ctx]), NBG)]
        v = {}
        for s in subs:
            X = smp.copy()
            for k in s:
                X[:, k] = g[i, k]
            ok = np.ones(len(X), bool)
            for k, gn in enumerate(bd[ctx]):
                lo, hi = bd[ctx][gn]
                ok &= (X[:, k] >= lo) & (X[:, k] <= hi)
            v[s] = ok.mean()
        for k in range(4):
            tot = 0.0
            for s in subs:
                if k in s:
                    continue
                w = factorial(len(s)) * factorial(3 - len(s)) / factorial(4)
                tot += w * (v[tuple(sorted(s + (k,)))] - v[s])
            S[n, k] = tot
    return S


gate, probe, expl_stats = [], [], []
for dom in ["RLV", "Healthcare"]:
    cfg = CFG[dom]
    b, g, q, ts = load(dom)
    sl, bd = slacks(b, g, cfg)
    Ag = b.A_g.values.astype(bool)
    Disc = (q >= QMIN).all(axis=1)
    AR = Ag & Disc
    Aimm = AR & (ts >= TAU)
    ev = pd.DataFrame({"event_id": b.event_id, "AR": AR, "Aimm": Aimm}) \
           .groupby("event_id")[["AR", "Aimm"]].max()
    gate.append(dict(domain=dom, n=len(b), Ag=Ag.mean(), Disc=Disc.mean(),
                     AR=AR.mean(), TS_pass=(ts >= TAU).mean(), Aimm=Aimm.mean(),
                     blocked_by_disclosure=(Ag & ~AR).mean(),
                     blocked_by_immune=(AR & ~Aimm).mean(),
                     eq215_violations=int((AR & ~Ag).sum()),
                     event_level_AR=ev.AR.mean(), event_level_Aimm=ev.Aimm.mean()))

    idx = np.where(Aimm)[0]
    marg = sl[idx].min(axis=1)
    bind = sl[idx].argmin(axis=1)
    S = shapley(g, b, bd, idx, cfg)
    sh = np.abs(S).argmax(axis=1)
    row = dict(domain=dom, n_released=len(idx),
               margin_median=float(np.median(marg)),
               within_005=float((marg < 0.05).mean()),
               agreement_analytic_vs_shapley=float((bind == sh).mean()))
    for k in range(4):
        row[f"median_displacement_{NAMES[dom][k]}"] = float(np.median(sl[idx][:, k]))
        row[f"binding_share_{NAMES[dom][k]}"] = float((bind == k).mean())
        row[f"shapley_top_share_{NAMES[dom][k]}"] = float((sh == k).mean())
    probe.append(row)

    payloads = [json.dumps({"binding_coordinate": NAMES[dom][int(bind[n])],
                            "slack": round(float(marg[n]), 4),
                            "coverage_min": round(float(q[i].min()), 3),
                            "transition_stability": round(float(ts[i]), 4),
                            "context_label": b.context_label[i],
                            "regime_version": "C1", "route": "proceed"},
                           sort_keys=True, separators=(",", ":"))
                for n, i in enumerate(idx)]
    dig = {hashlib.sha256(p.encode()).hexdigest() for p in payloads}
    L = [len(p) for p in payloads]
    w = np.array([[bd[c][gn][1] - bd[c][gn][0] for gn in bd[c]] for c in b.context_label[idx]])
    free = np.prod(np.clip(1 - 2 * marg[:, None] / w, 0, 1), axis=1) / \
           np.clip(1 - 2 * marg / w[:, 0], 1e-9, None)
    expl_stats.append(dict(domain=dom, n_released=len(idx),
                           payload_bytes_median=int(np.median(L)),
                           payload_bytes_min=int(min(L)), payload_bytes_max=int(max(L)),
                           distinct_digests=len(dig),
                           consistent_box_fraction_median=float(np.median(free))))

pd.DataFrame(gate).to_csv(D + "release_gating.csv", index=False)
pd.DataFrame(probe).to_csv(D + "counterfactual_probe.csv", index=False)
pd.DataFrame(expl_stats).to_csv(D + "expl_payload_stats.csv", index=False)
for t, f in [("release gating", "release_gating.csv"),
             ("counterfactual probe", "counterfactual_probe.csv"),
             ("explanation payload", "expl_payload_stats.csv")]:
    print(f"Saved: data/{f}  ({t})")
