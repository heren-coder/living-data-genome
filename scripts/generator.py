"""
Living Data Genome — scenario generator
Domain-agnostic synthetic scenario generator + simplified 4-gene mapping.

Design goals:
  - Domain-agnostic core: the SAME generator code drives both the RLV testbed
    and the healthcare access-governance vignette (Section 4.8), via a
    DomainConfig object. Nothing RLV-specific is hardcoded in the core.
  - Simplified trace-to-gene proxy Pi(.) : MLP + LoRA-style context-conditioned
    adaptation. It is untrained, and it stands in for a transformer-based
    encoder (Vaswani et al., 2017) only in the architectural sense of a
    shared base map specialized per context through a low-rank correction.
  - Bounded genesis: each incident produces a FAMILY of K admissible candidate
    seeds (Eq. 2.12), not a single point — needed later for SR (survival rate)
    and ablation.
  - context_label (c) is attached to every event from the start (cannot be
    added retroactively — needed for Section 2.5.7 ectopic-expression screening
    by day5_ectopic.py).
  - pi(c): protocol-declared (not learned) expected gene-dominance profile per
    context, used later to compute Xi_i,t(c) (Eq. A14).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

RNG_SEED = 42


# ---------------------------------------------------------------------------
# 1. Domain configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainConfig:
    name: str                                  # "RLV" or "Healthcare"
    gene_labels: Dict[str, str]                # abstract {S,A,D,E} -> domain-specific name
    context_categories: List[str]              # discrete context labels c in C
    context_means: Dict[str, np.ndarray]       # per-context latent regime mean (drives raw trace)
    raw_trace_dim: int                         # dimensionality of the raw (partial) trace u_{i,t}
    feasibility_tolerance: float               # G(C_i,t): context-conditioned band half-width around regime mean
    pi: Dict[str, np.ndarray]                  # pi(c): protocol-declared expected dominance profile

    def feasibility_bounds(self, context: str) -> Dict[str, Tuple[float, float]]:
        """
        G(C_i,t): admissible region induced by the ACTIVE constraint regime.
        Deliberately context-conditioned (not a fixed [0,1] box) and
        deliberately independent of pi(c)/A_pi(.) (Section 2.5.7): this
        tests raw physical/causal feasibility, not dominance-profile shape.
        """
        mean = self.context_means[context]
        bounds = {}
        for i, k in enumerate(["S", "A", "D", "E"]):
            lo = max(0.0, mean[i] - self.feasibility_tolerance)
            hi = min(1.0, mean[i] + self.feasibility_tolerance)
            bounds[k] = (lo, hi)
        return bounds


def _simplex(v: np.ndarray) -> np.ndarray:
    """Project a non-negative vector onto the probability simplex (sum=1)."""
    v = np.clip(v, 1e-6, None)
    return v / v.sum()


# --- RLV domain -------------------------------------------------------------

RLV_CONTEXTS = ["dense_day", "sparse_day", "dense_night", "sparse_night"]

RLV_CONTEXT_MEANS = {
    # latent regime mean over [speed, attention, density, environment] raw drivers
    # NOTE on Speed: values are grounded in RLV-specific reasoning about the
    # violation moment itself (not merely background traffic flow). In
    # sparse/night conditions there is less physical obstruction and less
    # social/visibility deterrence, so entry speed at a violation tends to be
    # higher; in dense/day conditions, congestion physically caps speed and
    # visibility increases deterrence. This widens Speed's cross-context span
    # to be comparable to Density's and Environment's, rather than leaving it
    # as a narrow outlier reflecting only average traffic-flow speed.
    "dense_day":    np.array([0.35, 0.55, 0.85, 0.30]),
    "sparse_day":   np.array([0.60, 0.60, 0.20, 0.25]),
    "dense_night":  np.array([0.40, 0.35, 0.75, 0.70]),
    "sparse_night": np.array([0.80, 0.30, 0.15, 0.80]),
}

# pi(c): protocol-declared expected dominance profile (S,A,D,E), sums to 1.
# Rule-based, NOT learned — e.g. dense/day contexts expect Density+Attention to
# dominate; sparse/night contexts expect Speed+Environment to dominate.
RLV_PI = {
    "dense_day":    _simplex(np.array([0.20, 0.30, 0.35, 0.15])),
    "sparse_day":   _simplex(np.array([0.35, 0.25, 0.15, 0.25])),
    "dense_night":  _simplex(np.array([0.20, 0.25, 0.30, 0.25])),
    "sparse_night": _simplex(np.array([0.35, 0.15, 0.10, 0.40])),
}

RLV_CONFIG = DomainConfig(
    name="RLV",
    gene_labels={"S": "Speed", "A": "Attention", "D": "Density", "E": "Environment"},
    context_categories=RLV_CONTEXTS,
    context_means=RLV_CONTEXT_MEANS,
    raw_trace_dim=8,
    feasibility_tolerance=0.33,
    pi=RLV_PI,
)


# --- Healthcare access-governance domain (Section 4.8) ----------------------
# Healthcare access-governance configuration, expressed in the framework's own
# vocabulary (gene coordinates, admissibility, context mismatch) rather than in
# domain-specific risk or architecture terms.

HC_CONTEXTS = ["high_load_daytime", "low_load_daytime", "high_load_nighttime", "low_load_nighttime"]

HC_CONTEXT_MEANS = {
    # latent regime mean over [access-escalation, audit-coverage, concurrent-density, regime-context]
    # NOTE on AccessEscalationRate: analogous to RLV's Speed, this is the
    # gene that defines the anomalous event itself (an escalating,
    # unauthorized access pattern), not merely a background load average.
    # Under low oversight (low audit coverage, off-hours, low staff
    # presence) an escalation can proceed far more freely; under high
    # daytime oversight it is comparatively contained. This widens the
    # cross-context span to be comparable to ConcurrentRecordDensity's,
    # rather than leaving it as a narrow outlier.
    "high_load_daytime":   np.array([0.25, 0.65, 0.80, 0.30]),
    "low_load_daytime":    np.array([0.35, 0.70, 0.25, 0.25]),
    "high_load_nighttime": np.array([0.50, 0.35, 0.70, 0.65]),
    "low_load_nighttime":  np.array([0.85, 0.30, 0.15, 0.75]),
}

HC_PI = {
    "high_load_daytime":   _simplex(np.array([0.15, 0.35, 0.35, 0.15])),
    "low_load_daytime":    _simplex(np.array([0.20, 0.40, 0.15, 0.25])),
    "high_load_nighttime": _simplex(np.array([0.25, 0.20, 0.30, 0.25])),
    "low_load_nighttime":  _simplex(np.array([0.35, 0.15, 0.10, 0.40])),
}

HEALTHCARE_CONFIG = DomainConfig(
    name="Healthcare",
    gene_labels={
        "S": "AccessEscalationRate",
        "A": "AuditCoverage",
        "D": "ConcurrentRecordDensity",
        "E": "AccessControlRegime",
    },
    context_categories=HC_CONTEXTS,
    context_means=HC_CONTEXT_MEANS,
    raw_trace_dim=8,
    feasibility_tolerance=0.31,
    pi=HC_PI,
)


# ---------------------------------------------------------------------------
# 2. Simplified trace-to-gene proxy: MLP + LoRA-style context adaptation
#    (untrained; stands in for a transformer-based encoder in the
#    architectural sense only)
# ---------------------------------------------------------------------------

class TraceToGeneEncoder:
    """
    A minimal MLP encoder Pi(.) : raw trace u_{i,t} -> gene coordinate g_{i,t}.

    Design: the gene coordinate is anchored to the context's latent regime
    mean (so that dominance patterns remain generally consistent with the
    declared context, as expected by a well-behaved proxy operator), and the
    MLP + LoRA-style context adaptation contributes a *bounded correction*
    on top of that anchor, refining the raw proxy rather than replacing it
    outright. It stands in for a transformer-based encoder in the
    architectural sense only (Section 3.12
    measures the dependence on that choice). The encoder is untrained: its weights are drawn once from a fixed
    seed and never updated.
    A minority of candidates will still drift enough (via mutation jitter,
    downstream) to be flagged by the context-mismatch screening in
    Section 2.5.7 — this is intentional, not a bug.
    """

    def __init__(self, raw_dim: int, contexts: List[str], rank: int = 2,
                 correction_scale: float = 0.18, seed: int = RNG_SEED):
        rng = np.random.default_rng(seed)
        self.raw_dim = raw_dim
        self.rank = rank
        self.correction_scale = correction_scale
        self.W0 = rng.normal(0, 0.3, size=(raw_dim, 4))
        self.b0 = rng.normal(0, 0.05, size=(4,))
        self.lora = {}
        for c in contexts:
            B = rng.normal(0, 0.25, size=(raw_dim, rank))
            A = rng.normal(0, 0.25, size=(rank, 4))
            self.lora[c] = B @ A  # low-rank context-specific delta, same shape as W0

    def encode(self, u: np.ndarray, context: str, anchor: np.ndarray) -> np.ndarray:
        W = self.W0 + self.lora[context]
        correction = np.tanh(u @ W + self.b0) * self.correction_scale
        g = np.clip(anchor + correction, 0.0, 1.0)
        return g


# ---------------------------------------------------------------------------
# 3. Bounded genesis + mutation + repair: a real 3-stage trajectory per
#    candidate (Eq. 2.11-2.15), not a single static snapshot.
#
#    This is what makes the "lifecycle" claim operational rather than
#    asserted: repair's effect (recovering admissibility lost during
#    mutation) becomes a MEASURABLE property of the generated data, checked
#    by rel_computation.py, rather than a name attached to unchanging data.
# ---------------------------------------------------------------------------

def generate_domain_dataset(
    config: DomainConfig,
    n_events: int = 200,
    k_candidates_range: Tuple[int, int] = (3, 6),
    trace_noise: float = 0.12,
    candidate_jitter: float = 0.05,
    mutation_spread: float = 0.14,
    repair_pull: float = 0.30,
    missing_rate: float = 0.15,
    independent_gene_noise_std: float = 0.18,
    seed: int = RNG_SEED,
    trace_sink: "list | None" = None,
) -> pd.DataFrame:
    """
    Generate n_events post-incident scenarios for a given domain. Each event:
      - is assigned a context_label c (protocol-relevant regime)
      - produces a raw partial trace u_{i,t} (context-conditioned, noisy, with
        missingness -> quality metadata q)
      - is mapped through Pi(.) into a genesis candidate FAMILY of K feasible
        seeds E_i,t^(0,k) (Eq. 2.12)
      - EACH candidate then follows a 3-stage trajectory:
          stage_t=0  genesis  (Eq. 2.11-2.12): anchored, bounded reconstruction
          stage_t=1  mutation (Eq. 2.13-2.14): wider, less constrained
                     exploration -- may leave the admissible region G(C)
          stage_t=2  repair    (Eq. 2.15): projection back toward the
                     admissible region's center -- "improvement over time"
                     is a measured recovery in admissibility, not an
                     asserted property

    NOTE on independent_gene_noise_std: the context-mean anchor alone was
    found to induce strong artificial inter-gene correlation (|r|=0.69-0.93
    across all six gene pairs in both domains), since every gene traces back
    to the same shared per-context tuple. This is a construction artifact,
    not a claim that the four genes are causally redundant -- it directly
    undermines Section 2.3's K=4 independent-alphabet framing if left
    uncorrected. An independent, per-gene, event-level perturbation is added
    to the anchor BEFORE encoding, representing genuine within-context
    individual variation that a coarse 4-context label cannot capture (e.g.,
    two vehicles in the same "dense_day" regime still differ in speed for
    reasons unrelated to density). This decorrelates the four dimensions
    substantially while preserving the context's role as an expected
    regime, not a deterministic determinant.
    """
    rng = np.random.default_rng(seed)
    encoder = TraceToGeneEncoder(config.raw_trace_dim, config.context_categories, seed=seed)

    rows = []
    for event_id in range(n_events):
        context = rng.choice(config.context_categories)
        regime_mean = config.context_means[context]  # length-4 latent drivers
        bounds = config.feasibility_bounds(context)
        lo = np.array([bounds[k][0] for k in ["S", "A", "D", "E"]])
        hi = np.array([bounds[k][1] for k in ["S", "A", "D", "E"]])
        region_center = (lo + hi) / 2.0

        # independent, gene-decorrelated within-context individual variation
        # (see docstring note) -- applied to the anchor itself, before any
        # shared encoding, so it is not re-correlated by the encoder
        gene_indep_noise = rng.normal(0, independent_gene_noise_std, size=4)
        anchor = np.clip(regime_mean + gene_indep_noise, 0.0, 1.0)

        # expand latent 4-dim regime mean into a raw_trace_dim vector via a
        # fixed random projection + noise, simulating partial sensor/log traces
        proj = np.tile(regime_mean, config.raw_trace_dim // 4 + 1)[: config.raw_trace_dim]
        u = proj + rng.normal(0, trace_noise, size=config.raw_trace_dim)

        # The protected trace u is the DNA-level view of the cross-layer
        # coherence proxy (Eq. 2.25). It is recorded here only; no random
        # draw is added, so scenarios.csv remains bit-identical.
        if trace_sink is not None:
            trace_sink.append(
                {"domain": config.name, "event_id": event_id, "context_label": context,
                 **{f"u_{d}": u[d] for d in range(config.raw_trace_dim)}}
            )

        # quality metadata: simulate per-gene coverage/reliability (missingness)
        q = np.clip(1.0 - rng.exponential(missing_rate, size=4), 0.05, 1.0)

        k_family = int(rng.integers(k_candidates_range[0], k_candidates_range[1] + 1))
        base_g = encoder.encode(u, context, anchor)

        for cand_idx in range(k_family):
            # t=0: genesis -- bounded reconstruction, small jitter around the
            # anchored encoding (this preserves the "K admissible seeds" family)
            g0 = np.clip(base_g + rng.normal(0, candidate_jitter, size=4), 0.0, 1.0)

            # t=1: mutation -- wider, less constrained exploration; may leave G(C)
            g1 = np.clip(g0 + rng.normal(0, mutation_spread, size=4), 0.0, 1.0)

            # t=2: repair -- projection toward the admissible region's center,
            # plus a small residual perturbation (repair is a projection, not
            # a perfect fix)
            g2 = np.clip(
                g1 + repair_pull * (region_center - g1) + rng.normal(0, mutation_spread * 0.3, size=4),
                0.0, 1.0,
            )

            for stage_t, g in [(0, g0), (1, g1), (2, g2)]:
                admissible = all(
                    bounds[k_][0] <= g[i] <= bounds[k_][1]
                    for i, k_ in enumerate(["S", "A", "D", "E"])
                )
                rows.append({
                    "domain": config.name,
                    "event_id": event_id,
                    "candidate_id": cand_idx,
                    "stage_t": stage_t,
                    "context_label": context,
                    "g_S": g[0], "g_A": g[1], "g_D": g[2], "g_E": g[3],
                    "q_S": q[0], "q_A": q[1], "q_D": q[2], "q_E": q[3],
                    "A_g": admissible,
                    "n_candidates_in_family": k_family,
                })

    df = pd.DataFrame(rows)
    return df


def pi_lookup_frame(config: DomainConfig) -> pd.DataFrame:
    """Return pi(c) as a tidy dataframe for the Xi_i,t(c) computation of day5_ectopic.py."""
    rows = []
    for c, vec in config.pi.items():
        rows.append({"domain": config.name, "context_label": c,
                      "pi_S": vec[0], "pi_A": vec[1], "pi_D": vec[2], "pi_E": vec[3]})
    return pd.DataFrame(rows)


OUT = "../data/"

if __name__ == "__main__":
    trace_sink = []
    rlv_df = generate_domain_dataset(RLV_CONFIG, n_events=200, seed=RNG_SEED, trace_sink=trace_sink)
    hc_df = generate_domain_dataset(HEALTHCARE_CONFIG, n_events=200, seed=RNG_SEED + 1, trace_sink=trace_sink)
    pd.DataFrame(trace_sink).to_csv(OUT + "raw_traces.csv", index=False)

    full_df = pd.concat([rlv_df, hc_df], ignore_index=True)
    full_df.to_csv(OUT + "scenarios.csv", index=False)

    pi_df = pd.concat([pi_lookup_frame(RLV_CONFIG), pi_lookup_frame(HEALTHCARE_CONFIG)], ignore_index=True)
    pi_df.to_csv(OUT + "pi_lookup.csv", index=False)

    print("=== Generation summary ===")
    for dname, d in [("RLV", rlv_df), ("Healthcare", hc_df)]:
        print(f"\n--- {dname} ---")
        print(f"events: {d['event_id'].nunique()}, candidate-trajectories: {d['candidate_id'].count() // 3}")
        print("admissibility by lifecycle stage (0=genesis, 1=mutation, 2=repair):")
        print(d.groupby("stage_t")["A_g"].mean().round(3))
        print("context distribution:")
        print(d[d.stage_t == 0].groupby("context_label")["event_id"].nunique())

    print("\n=== pi(c) lookup table ===")
    print(pi_df.round(3))

    print("\n=== Sanity check: realized dominance vs pi(c) (RLV, genesis stage) ===")
    g0 = rlv_df[rlv_df.stage_t == 0]
    for c in RLV_CONTEXTS:
        sub = g0[g0.context_label == c][["g_S", "g_A", "g_D", "g_E"]].mean().values
        realized_dom = _simplex(sub)
        expected = RLV_PI[c]
        print(f"{c:15s} realized_dom={np.round(realized_dom,3)}  pi(c)={np.round(expected,3)}")

    print(f"\nSaved: {OUT}raw_traces.csv, {OUT}scenarios.csv, {OUT}pi_lookup.csv")
