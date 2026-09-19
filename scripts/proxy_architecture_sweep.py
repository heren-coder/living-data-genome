"""
proxy_architecture_sweep.py -- sensitivity of Rel and its components to the
architecture of the trace-to-gene proxy Pi.

Why this script exists
----------------------
The manuscript relates Pi to a transformer-based encoder (Vaswani et al.,
2017); the experiments use TraceToGeneEncoder, a fixed random-feature MLP
(random projection + tanh + LoRA-style context deltas, no training objective)
that stands in for it in the architectural sense only. The question is what
this choice implies for the reported conclusions. Since the framework defines no learning objective for Pi, a
trained transformer cannot be substituted without inventing one; the
relevant question is instead whether the results depend on the proxy's
architecture WITHIN the random-feature regime. This script varies that
architecture and nothing else.

Matched design: the generator's own RNG stream (contexts, anchors, traces,
quality, candidate jitter, mutation, repair) is untouched -- only
encoder.encode() changes -- so for a given generator seed every arm sees
the same events and differs only in base_g. The reference arm reproduces
the submitted data exactly (verified at seed 0).

Arms (all untrained, all bounded by tanh * correction_scale):
  mlp1_tanh_s018_r2 : reference (submitted)
  mlp1_relu         : ReLU hidden nonlinearity (output still tanh-bounded)
  mlp2_tanh         : two hidden layers (8 -> 8 -> 4)
  mlp1_s009 / s036  : correction_scale halved / doubled
  mlp1_r1 / r4      : LoRA rank 1 / 4
  attn1             : single-head random-feature attention over the trace
                      read as 2 tokens of 4 dims, mean-pooled, then the same
                      output map + LoRA delta as the reference
Rel is computed with compute_rel (GC placeholder = 1, matched evaluation
seed) on N_SEEDS generator seeds per arm. Two diagnostics complete the
table: Pearson r between arm and reference base_g displacements (how much
the geometry moves) and the fraction of candidates whose repair-stage
admissibility A_g agrees with the reference (how much the gate moves).

Outputs: data/multiseed/proxy_arch_sweep.csv          (per arm x seed x domain)
         data/multiseed/proxy_arch_sweep_summary.csv  (mean, SD, 95 % CI; delta vs reference)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import generator as G                                         # noqa: E402
from generator import RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG  # noqa: E402
from rel_computation import compute_rel                        # noqa: E402

D = os.path.join(_HERE, "..", "data", "multiseed") + os.sep
CONFIG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}
N_SEEDS = int(os.environ.get("LDG_N_SEEDS", 50))
SEED_START = int(os.environ.get("LDG_SEED_START", 0))   # chunked runs: LDG_SEED_START/LDG_N_SEEDS, then LDG_SUMMARY_ONLY=1
SUMMARY_ONLY = os.environ.get("LDG_SUMMARY_ONLY", "0") == "1"
STRIDE = 1000
B_BOOT = 10000
BRNG = np.random.default_rng(RNG_SEED + 2028)


def boot_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[BRNG.integers(0, len(x), size=(B_BOOT, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ---------------------------------------------------------------- encoder arms
class _Base(G.TraceToGeneEncoder):
    """Keeps W0, b0, lora exactly as the reference so arms differ only where stated."""
    act = "tanh"
    scale = 0.18

    def __init__(self, raw_dim, contexts, rank=2, correction_scale=None, seed=RNG_SEED):
        super().__init__(raw_dim, contexts, rank=rank,
                         correction_scale=self.scale if correction_scale is None else correction_scale, seed=seed)
        self._extra_rng = np.random.default_rng(seed + 555)   # extra weights, drawn AFTER the reference ones

    def hidden(self, u, context):
        return u @ (self.W0 + self.lora[context]) + self.b0

    def encode(self, u, context, anchor):
        h = self.hidden(u, context)
        return np.clip(anchor + np.tanh(h) * self.correction_scale, 0.0, 1.0)


class MLP1Tanh(_Base):
    pass


class MLP1ReLU(_Base):
    def encode(self, u, context, anchor):
        h = np.maximum(self.hidden(u, context), 0.0)
        return np.clip(anchor + np.tanh(h) * self.correction_scale, 0.0, 1.0)


class MLP2Tanh(_Base):
    def __init__(self, raw_dim, contexts, **kw):
        super().__init__(raw_dim, contexts, **kw)
        self.W1 = self._extra_rng.normal(0, 0.3, size=(raw_dim, raw_dim))
        self.b1 = self._extra_rng.normal(0, 0.05, size=(raw_dim,))

    def hidden(self, u, context):
        h1 = np.tanh(u @ self.W1 + self.b1)
        return h1 @ (self.W0 + self.lora[context]) + self.b0


class Attn1(_Base):
    """Single-head random-feature attention: trace u (8) read as 2 tokens x 4 dims."""
    def __init__(self, raw_dim, contexts, **kw):
        super().__init__(raw_dim, contexts, **kw)
        self.dtok, self.ntok = 4, raw_dim // 4
        r = self._extra_rng
        self.Wq = r.normal(0, 0.3, size=(self.dtok, self.dtok))
        self.Wk = r.normal(0, 0.3, size=(self.dtok, self.dtok))
        self.Wv = r.normal(0, 0.3, size=(self.dtok, self.dtok))
        self.Wo = r.normal(0, 0.3, size=(self.dtok, raw_dim))   # back to raw_dim so W0/LoRA reuse

    def hidden(self, u, context):
        X = u.reshape(self.ntok, self.dtok)
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        A = Q @ K.T / np.sqrt(self.dtok)
        A = np.exp(A - A.max(axis=1, keepdims=True)); A /= A.sum(axis=1, keepdims=True)
        Z = (A @ V).mean(axis=0) @ self.Wo                           # pooled, raw_dim
        return Z @ (self.W0 + self.lora[context]) + self.b0


def make(cls, **fixed):
    def factory(raw_dim, contexts, rank=2, correction_scale=0.18, seed=RNG_SEED):
        kw = dict(rank=rank, correction_scale=correction_scale, seed=seed); kw.update(fixed)
        return cls(raw_dim, contexts, **kw)
    return factory


ARMS = {
    "mlp1_tanh_s018_r2": make(MLP1Tanh),                       # reference
    "mlp1_relu":         make(MLP1ReLU),
    "mlp2_tanh":         make(MLP2Tanh),
    "mlp1_s009":         make(MLP1Tanh, correction_scale=0.09),
    "mlp1_s036":         make(MLP1Tanh, correction_scale=0.36),
    "mlp1_r1":           make(MLP1Tanh, rank=1),
    "mlp1_r4":           make(MLP1Tanh, rank=4),
    "attn1":             make(Attn1),
}
REF = "mlp1_tanh_s018_r2"
GCOLS = ["g_S", "g_A", "g_D", "g_E"]


def generate_with(arm, k):
    G.TraceToGeneEncoder = ARMS[arm]      # generate_domain_dataset resolves the name at call time
    try:
        out = {}
        for dom, cfg in CONFIG.items():
            s = RNG_SEED + STRIDE * k + (0 if dom == "RLV" else 1)
            out[dom] = (G.generate_domain_dataset(cfg, n_events=200, seed=s), s)
        return out
    finally:
        G.TraceToGeneEncoder = _ORIG


_ORIG = G.TraceToGeneEncoder

if __name__ == "__main__":
    t0 = time.time()
    # sanity: reference arm reproduces submitted data at seed 0
    ref0 = generate_with(REF, 0)
    sub0 = pd.read_csv(os.path.join(_HERE, "..", "data", "scenarios.csv"))
    ok = all(np.allclose(ref0[d][0][GCOLS].values, sub0[sub0.domain == d][GCOLS].values, atol=1e-12) for d in CONFIG)
    print("reference arm reproduces scenarios.csv at seed 0:", "PASS" if ok else "FAIL")

    rows = []
    for k in ([] if SUMMARY_ONLY else range(SEED_START, SEED_START + N_SEEDS)):
        ref = generate_with(REF, k)
        for arm in ARMS:
            data = ref if arm == REF else generate_with(arm, k)
            for dom, cfg in CONFIG.items():
                df, s = data[dom]
                summ, per_event, _ = compute_rel(dom, df, cfg, seed=s)
                rdf = ref[dom][0]
                gen, rgen = df[df.stage_t == 0][GCOLS].values, rdf[rdf.stage_t == 0][GCOLS].values
                rep, rrep = df[df.stage_t == 2], rdf[rdf.stage_t == 2]
                disp = (gen - rgen).ravel()
                rows.append({
                    "arm": arm, "k": k, "domain": dom,
                    "A_triad": summ["A_triad"], "TS_mean": summ["TS_mean"], "SR_mean": summ["SR_mean"],
                    "Rel_mean": summ["Rel_mean"],
                    "genesis_rmsd_vs_ref": float(np.sqrt((disp ** 2).mean())),
                    "Ag_agreement_vs_ref": float((rep["A_g"].values == rrep["A_g"].values).mean()),
                })
        if k % 10 == 9:
            print(f"  seed {k + 1}/{N_SEEDS} ({time.time() - t0:.0f}s)")
    part = D + "proxy_arch_sweep.csv"
    if rows:
        new = pd.DataFrame(rows)
        if SEED_START > 0 and os.path.exists(part):
            old = pd.read_csv(part); old = old[~old.k.isin(new.k.unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(part, index=False)
    res = pd.read_csv(part).sort_values(["k", "arm", "domain"]).reset_index(drop=True)
    N_SEEDS = int(res.k.nunique())

    # summary with paired deltas vs reference
    ref = res[res.arm == REF].set_index(["k", "domain"])
    out = []
    for (arm, dom), g in res.groupby(["arm", "domain"]):
        row = {"arm": arm, "domain": dom, "n_seeds": int(g.k.nunique())}
        for c in ["A_triad", "TS_mean", "SR_mean", "Rel_mean", "genesis_rmsd_vs_ref", "Ag_agreement_vs_ref"]:
            lo, hi = boot_ci(g[c]); row[f"{c}_mean"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = float(g[c].mean()), lo, hi
        for c in ["A_triad", "TS_mean", "SR_mean", "Rel_mean"]:
            d = g.set_index(["k", "domain"])[c] - ref.loc[g.set_index(["k", "domain"]).index, c]
            lo, hi = boot_ci(d); row[f"d{c}_mean"], row[f"d{c}_ci_lo"], row[f"d{c}_ci_hi"] = float(d.mean()), lo, hi
        out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(D + "proxy_arch_sweep_summary.csv", index=False)

    print(f"\n=== proxy architecture sweep, {N_SEEDS} seeds, {time.time() - t0:.0f}s ===")
    for dom in CONFIG:
        print(f"\n--- {dom} ---  (paired delta vs reference, mean [95% CI])")
        o = out[out.domain == dom].set_index("arm").loc[list(ARMS)]
        for arm, r in o.iterrows():
            print(f"  {arm:<18} Rel {r.Rel_mean_mean:.4f}  dRel {r.dRel_mean_mean:+.4f} [{r.dRel_mean_ci_lo:+.4f},{r.dRel_mean_ci_hi:+.4f}]"
                  f"  dA {r.dA_triad_mean:+.4f}  dTS {r.dTS_mean_mean:+.4f}  dSR {r.dSR_mean_mean:+.4f}"
                  f"  gRMSD {r.genesis_rmsd_vs_ref_mean:.3f}  Ag-agree {r.Ag_agreement_vs_ref_mean:.3f}")
    print(f"\nSaved: {D}proxy_arch_sweep.csv, proxy_arch_sweep_summary.csv")
