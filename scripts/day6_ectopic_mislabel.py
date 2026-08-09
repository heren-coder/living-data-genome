"""
A1 -- Section 2.5.7 validation: context-label misassignment.

Section 2.5.7 makes three claims that the routing/repair demonstration in
day5_ectopic.py does not test:

  (i)   the screen A_pi = 1[Xi <= xi] detects a context mismatch;
  (ii)  the separation test A_ctx = 1[Xi(c) - Xi(c*) >= eta] attributes the
        mismatch to the declared LABEL rather than to the reconstruction;
  (iii) repairing a candidate whose label (not reconstruction) was wrong
        pulls its dominance profile toward an expectation that does not
        hold, and therefore corrupts admissible evidence.

Claim (iii) must NOT be tested by comparing Xi against the true label
before vs after repair. That comparison is confounded: repair is first of
all a denoising step, and pulling a noisy gene coordinate toward ANY
context anchor moves it closer to the true anchor than it started. The
label-correctness signal is swamped by the denoising gain, and the naive
test can even show an apparent improvement under a wrong label.

The correct test is matched: the SAME candidate is repaired twice, once
toward the declared (wrong) profile and once toward the true profile. The
difference is the cost of the misrouting itself, with the denoising gain
present in both arms and therefore cancelled.

Design: at the repair stage (t=2), a fraction eps of candidates has its
declared context label replaced by a different label from the same domain.
The true label is retained only for scoring. Nothing downstream sees it.

Measured:
  detection_rate        P(A_pi = 0 | mislabeled)          screen fires
  false_flag_rate       P(A_pi = 0 | correctly labeled)   screen fires spuriously
  attribution_rate      P(A_ctx = 1 | mislabeled, flagged)
  misattribution_rate   P(A_ctx = 1 | correct label, flagged)
  label_recovery_rate   P(c* = true label | mislabeled, flagged, A_ctx = 1)
  corruption            Xi against the TRUE label, before vs after repair,
                        for mislabeled candidates the router sent to repair
                        (retained for continuity; confounded, see above)
  misrouting_cost       matched design: Xi against the TRUE label after
                        repair toward the declared profile minus the same
                        quantity after repair toward the true profile
  ctx_distance          JSD between the declared and the true context
                        profiles, split by whether the separation test
                        attributed the mismatch
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, _simplex
from day5_ectopic import context_repair

SEEDS = [0, 1, 2, 3, 4]
MATCHED_SEEDS = list(range(40))   # matched design: the misrouted cell is a rare
                                  # intersection, so it needs more draws to fill
XI = 0.010          # reference tolerance, matches Table 3.6
ETA = 0.010         # reference separation margin
EPS = 0.20          # reference misassignment rate
PULL = 0.40         # context-repair pull fraction, shared by both arms
CONFIG = {"RLV": RLV_CONFIG, "Healthcare": HEALTHCARE_CONFIG}


def _pi_map(pi_lookup, domain):
    return {r["context_label"]: np.array([r["pi_S"], r["pi_A"], r["pi_D"], r["pi_E"]])
            for _, r in pi_lookup[pi_lookup.domain == domain].iterrows()}


def _jsd(p, q):
    return float(jensenshannon(p, q) ** 2)


def build_trial(df, pi_lookup, domain, eps, seed):
    """Repair-stage candidates with a fraction eps relabelled."""
    rng = np.random.default_rng(seed)
    pim = _pi_map(pi_lookup, domain)
    labels = sorted(pim.keys())

    d = df[(df.domain == domain) & (df.stage_t == 2)].copy().reset_index(drop=True)
    d["true_context"] = d["context_label"]

    mis = rng.random(len(d)) < eps
    declared = []
    for i, row in d.iterrows():
        if mis[i]:
            alts = [c for c in labels if c != row["true_context"]]
            declared.append(rng.choice(alts))
        else:
            declared.append(row["true_context"])
    d["declared_context"] = declared
    d["mislabeled"] = mis

    rho = np.vstack([_simplex(r) for r in d[["g_S", "g_A", "g_D", "g_E"]].values])
    d["Xi_declared"] = [_jsd(rho[i], pim[d.declared_context[i]]) for i in range(len(d))]
    d["Xi_true"] = [_jsd(rho[i], pim[d.true_context[i]]) for i in range(len(d))]

    # best-explaining context and its divergence
    best_c, best_x = [], []
    for i in range(len(d)):
        xs = {c: _jsd(rho[i], pim[c]) for c in labels}
        c_star = min(xs, key=xs.get)
        best_c.append(c_star)
        best_x.append(xs[c_star])
    d["c_star"] = best_c
    d["Xi_star"] = best_x
    d["rho"] = list(rho)
    return d


def score(d, xi, eta):
    d = d.copy()
    d["A_pi"] = (d["Xi_declared"] <= xi).astype(int)
    d["A_ctx"] = ((d["Xi_declared"] - d["Xi_star"]) >= eta).astype(int)

    feas = d[d.A_g].copy()          # screen only applies to feasible candidates
    mis = feas[feas.mislabeled]
    cor = feas[~feas.mislabeled]

    flagged_mis = mis[mis.A_pi == 0]
    flagged_cor = cor[cor.A_pi == 0]

    out = {
        "n_feasible": len(feas),
        "n_mislabeled": len(mis),
        "detection_rate": float((mis.A_pi == 0).mean()) if len(mis) else np.nan,
        "false_flag_rate": float((cor.A_pi == 0).mean()) if len(cor) else np.nan,
        "attribution_rate": float((flagged_mis.A_ctx == 1).mean()) if len(flagged_mis) else np.nan,
        "misattribution_rate": float((flagged_cor.A_ctx == 1).mean()) if len(flagged_cor) else np.nan,
    }
    recov = flagged_mis[flagged_mis.A_ctx == 1]
    out["label_recovery_rate"] = float((recov.c_star == recov.true_context).mean()) if len(recov) else np.nan
    # the dangerous case: mislabeled, flagged, but NOT attributed -> sent to repair
    out["missed_attribution_n"] = int((flagged_mis.A_ctx == 0).sum())
    return out, d


def corruption(d, pi_lookup, domain, xi, eta, pull=0.4):
    """
    Claim (iii): mislabeled candidates that the separation test fails to
    catch are routed to repair, which pulls them toward the WRONG pi(c).
    Measured as Xi against the TRUE label, before vs after repair.
    """
    pim = _pi_map(pi_lookup, domain)
    cfg = CONFIG[domain]
    d = d.copy()
    d["A_pi"] = (d["Xi_declared"] <= xi).astype(int)
    d["A_ctx"] = ((d["Xi_declared"] - d["Xi_star"]) >= eta).astype(int)

    victims = d[d.A_g & d.mislabeled & (d.A_pi == 0) & (d.A_ctx == 0)]
    controls = d[d.A_g & (~d.mislabeled) & (d.A_pi == 0)]

    def repair_delta(sub):
        before, after = [], []
        for _, r in sub.iterrows():
            g = np.array([r.g_S, r.g_A, r.g_D, r.g_E])
            bounds = cfg.feasibility_bounds(r.declared_context)
            g2 = context_repair(g, pim[r.declared_context], bounds, pull)
            before.append(_jsd(_simplex(g), pim[r.true_context]))
            after.append(_jsd(_simplex(g2), pim[r.true_context]))
        if not before:
            return np.nan, np.nan, np.nan, 0
        b, a = np.array(before), np.array(after)
        return b.mean(), a.mean(), float((a > b).mean()), len(b)

    vb, va, vworse, vn = repair_delta(victims)
    cb, ca, cworse, cn = repair_delta(controls)
    return {
        "n_corrupted_route": vn,
        "Xi_true_before_repair": vb,
        "Xi_true_after_repair": va,
        "frac_worsened": vworse,
        "n_control": cn,
        "control_Xi_true_before": cb,
        "control_Xi_true_after": ca,
        "control_frac_worsened": cworse,
    }


def misrouting_cost(df, pi_lookup, domain, xi, eta, seeds, pull=PULL):
    """
    Matched test of claim (iii).

    For every candidate the router sends to repair under a wrong label,
    repair it twice from the same starting coordinate: once toward the
    declared profile and once toward the true profile. Both arms carry the
    same denoising gain, so their difference isolates the cost of acting on
    the wrong expectation.

    Candidates are deduplicated across seeds on (event, candidate, declared
    context); the same candidate drawn again under the same wrong label is
    the same observation, not a second one.
    """
    pim = _pi_map(pi_lookup, domain)
    cfg = CONFIG[domain]
    seen = set()
    before, wrong, right = [], [], []
    dist_attr, dist_missed = [], []

    for s in seeds:
        d = build_trial(df, pi_lookup, domain, EPS, s)
        d["A_pi"] = (d["Xi_declared"] <= xi).astype(int)
        d["A_ctx"] = ((d["Xi_declared"] - d["Xi_star"]) >= eta).astype(int)
        flagged_mis = d[d.A_g & d.mislabeled & (d.A_pi == 0)]

        for _, r in flagged_mis.iterrows():
            gap = _jsd(pim[r.declared_context], pim[r.true_context])
            (dist_attr if r.A_ctx == 1 else dist_missed).append(gap)
            if r.A_ctx == 1:
                continue                      # attributed: not sent to repair
            key = (r.event_id, r.candidate_id, r.declared_context)
            if key in seen:
                continue
            seen.add(key)
            g = np.array([r.g_S, r.g_A, r.g_D, r.g_E])
            gw = context_repair(g, pim[r.declared_context],
                                cfg.feasibility_bounds(r.declared_context), pull)
            gr = context_repair(g, pim[r.true_context],
                                cfg.feasibility_bounds(r.true_context), pull)
            before.append(_jsd(_simplex(g), pim[r.true_context]))
            wrong.append(_jsd(_simplex(gw), pim[r.true_context]))
            right.append(_jsd(_simplex(gr), pim[r.true_context]))

    b, w, rt = np.array(before), np.array(wrong), np.array(right)
    cost = w - rt
    _, p_matched = wilcoxon(rt, w)
    _, p_naive = wilcoxon(b, w)
    lo, hi = np.percentile(cost, [2.5, 97.5])

    summary = {
        "domain": domain,
        "n_misrouted": len(b),
        "n_seeds": len(seeds),
        "Xi_true_repaired_true_label": rt.mean(),
        "Xi_true_repaired_declared_label": w.mean(),
        "misrouting_cost_mean": cost.mean(),
        "misrouting_cost_median": float(np.median(cost)),
        "misrouting_cost_ci_lo": lo,
        "misrouting_cost_ci_hi": hi,
        "frac_cost_positive": float((cost > 0).mean()),
        "p_matched": p_matched,
        "Xi_true_before_repair": b.mean(),
        "frac_worsened_naive": float((w > b).mean()),
        "p_naive": p_naive,
        "ctx_gap_attributed": float(np.mean(dist_attr)),
        "ctx_gap_missed": float(np.mean(dist_missed)),
        "n_attributed": len(dist_attr),
        "n_missed_records": len(dist_missed),
    }
    pairs = pd.DataFrame({"domain": domain, "Xi_before": b,
                          "Xi_repair_true": rt, "Xi_repair_declared": w,
                          "cost": cost})
    return summary, pairs


def main():
    df = pd.read_csv("../data/scenarios.csv")
    pi_lookup = pd.read_csv("../data/pi_lookup.csv")

    # ---- reference operating point, 5 seeds -------------------------------
    print(f"=== A1 reference point (eps={EPS}, xi={XI}, eta={ETA}), mean +/- SD over {len(SEEDS)} seeds ===\n")
    ref_rows = []
    for domain in ["RLV", "Healthcare"]:
        per_seed, corr_seed = [], []
        for s in SEEDS:
            d = build_trial(df, pi_lookup, domain, EPS, s)
            m, _ = score(d, XI, ETA)
            per_seed.append(m)
            corr_seed.append(corruption(d, pi_lookup, domain, XI, ETA))
        agg = pd.DataFrame(per_seed)
        cagg = pd.DataFrame(corr_seed)
        print(f"--- {domain} ---")
        for k in ["detection_rate", "false_flag_rate", "attribution_rate",
                  "misattribution_rate", "label_recovery_rate"]:
            print(f"  {k:22s} {agg[k].mean():.3f} +/- {agg[k].std():.3f}")
        print(f"  {'n mislabeled':22s} {agg['n_mislabeled'].mean():.0f}")
        print(f"  {'n sent to repair':22s} {agg['missed_attribution_n'].mean():.0f}   (mislabeled, flagged, not attributed)")
        print(f"  corruption on those candidates, Xi against the TRUE label:")
        print(f"    before repair        {cagg['Xi_true_before_repair'].mean():.4f}")
        print(f"    after  repair        {cagg['Xi_true_after_repair'].mean():.4f}")
        print(f"    fraction worsened    {cagg['frac_worsened'].mean():.3f}")
        print(f"  control (correctly labelled, flagged, repaired):")
        print(f"    before / after       {cagg['control_Xi_true_before'].mean():.4f} / {cagg['control_Xi_true_after'].mean():.4f}")
        print(f"    fraction worsened    {cagg['control_frac_worsened'].mean():.3f}")
        print()
        r = {"domain": domain}
        r.update({k: agg[k].mean() for k in agg.columns})
        r.update({k + "_sd": agg[k].std() for k in agg.columns})
        r.update({k: cagg[k].mean() for k in cagg.columns})
        ref_rows.append(r)
    pd.DataFrame(ref_rows).to_csv("../data/a1_reference.csv", index=False)

    # ---- matched misrouting cost, claim (iii) ------------------------------
    print(f"=== Misrouting cost, matched design (eps={EPS}, xi={XI}, eta={ETA}), "
          f"{len(MATCHED_SEEDS)} seeds ===\n")
    cost_rows, pair_frames = [], []
    for domain in ["RLV", "Healthcare"]:
        s, pairs = misrouting_cost(df, pi_lookup, domain, XI, ETA, MATCHED_SEEDS)
        cost_rows.append(s)
        pair_frames.append(pairs)
        print(f"--- {domain} ---")
        print(f"  {'n misrouted (unique)':30s} {s['n_misrouted']}")
        print(f"  {'Xi after repair, true label':30s} {s['Xi_true_repaired_true_label']:.4f}")
        print(f"  {'Xi after repair, declared':30s} {s['Xi_true_repaired_declared_label']:.4f}")
        print(f"  {'misrouting cost, median':30s} {s['misrouting_cost_median']:+.5f}")
        print(f"  {'misrouting cost, mean':30s} {s['misrouting_cost_mean']:+.5f}")
        print(f"  {'95% interval':30s} [{s['misrouting_cost_ci_lo']:+.5f}, {s['misrouting_cost_ci_hi']:+.5f}]")
        print(f"  {'fraction with positive cost':30s} {s['frac_cost_positive']:.3f}")
        print(f"  {'Wilcoxon p (matched)':30s} {s['p_matched']:.3g}")
        print(f"  naive before/after, for contrast:")
        print(f"    {'Xi before repair':28s} {s['Xi_true_before_repair']:.4f}")
        print(f"    {'fraction worsened':28s} {s['frac_worsened_naive']:.3f}   p={s['p_naive']:.3g}")
        print(f"  context gap JSD(pi_declared, pi_true):")
        print(f"    {'attributed':28s} {s['ctx_gap_attributed']:.4f}  (n={s['n_attributed']})")
        print(f"    {'not attributed':28s} {s['ctx_gap_missed']:.4f}  (n={s['n_missed_records']})")
        print()
    pd.DataFrame(cost_rows).to_csv("../data/a1_misrouting_cost.csv", index=False)
    pd.concat(pair_frames).to_csv("../data/a1_misrouting_pairs.csv", index=False)

    # ---- eta sweep --------------------------------------------------------
    print("=== Separation margin sweep (eta), eps=%.2f, xi=%.3f ===" % (EPS, XI))
    rows = []
    for eta in [0.000, 0.002, 0.005, 0.010, 0.020, 0.030, 0.050]:
        for domain in ["RLV", "Healthcare"]:
            vals = []
            for s in SEEDS:
                d = build_trial(df, pi_lookup, domain, EPS, s)
                m, _ = score(d, XI, eta)
                vals.append(m)
            a = pd.DataFrame(vals)
            rows.append({"eta": eta, "domain": domain,
                         "attribution_rate": a.attribution_rate.mean(),
                         "misattribution_rate": a.misattribution_rate.mean(),
                         "label_recovery_rate": a.label_recovery_rate.mean()})
    eta_df = pd.DataFrame(rows)
    print(eta_df.round(3).to_string(index=False))
    eta_df.to_csv("../data/a1_eta_sweep.csv", index=False)

    # ---- eps sweep --------------------------------------------------------
    print("\n=== Misassignment rate sweep (eps), xi=%.3f, eta=%.3f ===" % (XI, ETA))
    rows = []
    for eps in [0.05, 0.10, 0.20, 0.30, 0.50]:
        for domain in ["RLV", "Healthcare"]:
            vals = []
            for s in SEEDS:
                d = build_trial(df, pi_lookup, domain, eps, s)
                m, _ = score(d, XI, ETA)
                vals.append(m)
            a = pd.DataFrame(vals)
            rows.append({"eps": eps, "domain": domain,
                         "detection_rate": a.detection_rate.mean(),
                         "false_flag_rate": a.false_flag_rate.mean(),
                         "attribution_rate": a.attribution_rate.mean(),
                         "misattribution_rate": a.misattribution_rate.mean()})
    eps_df = pd.DataFrame(rows)
    print(eps_df.round(3).to_string(index=False))
    eps_df.to_csv("../data/a1_eps_sweep.csv", index=False)

    print("\nSaved: a1_reference.csv, a1_misrouting_cost.csv, a1_misrouting_pairs.csv,\n       a1_eta_sweep.csv, a1_eps_sweep.csv")


if __name__ == "__main__":
    main()
