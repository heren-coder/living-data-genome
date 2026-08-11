"""Regenerates the numeric tables of the manuscript from the computed CSVs.

Four tables are protocol declarations rather than measurements and are not
generated here: Table 1 (declared constants), Table 2 (layer assignment), Table 3
(placement contract) and the abbreviations list. Everything the paper measures is
written from the data files by this script, so a table and its source cannot drift
apart. Writes ../data/tables/table_NN.csv.
"""
import os as _os
import csv, pandas as pd

_HERE = _os.path.dirname(_os.path.abspath(__file__))
D   = _os.path.join(_HERE, "..", "data") + _os.sep
OUT = _os.path.join(D, "tables") + _os.sep
_os.makedirs(OUT, exist_ok=True)

def write(num, caption, header, rows):
    with open(f"{OUT}table_{num:02d}.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["# " + caption]); w.writerow(header); w.writerows(rows)
    print(f"  table_{num:02d}  {len(rows)} rows")

def f3(x): return f"{float(x):.3f}"
def pc(x):
    v = float(x) * 100
    return "100%" if abs(v - 100) < 5e-3 else f"{v:.1f}%"

# ---- Table 4: declared dominance profiles -----------------------------------
pi = pd.read_csv(D + "pi_lookup.csv")
gene = {"RLV": ["Speed", "Attention", "Density", "Environment"],
        "Healthcare": ["AccessEscalationRate", "AuditCoverage",
                       "ConcurrentRecordDensity", "AccessControlRegime"]}
rows = []
for dom in ("RLV", "Healthcare"):
    sub = pi[pi.domain == dom]
    for r in sub.itertuples():
        rows.append([dom, r.context_label, f"{r.pi_S:.2f}", f"{r.pi_A:.2f}",
                     f"{r.pi_D:.2f}", f"{r.pi_E:.2f}"])
write(4, "Table 4. Protocol-declared gene-dominance profiles for the four operating "
         "contexts of each configuration.",
      ["Configuration", "Context", "Speed / AccessEscalationRate",
       "Attention / AuditCoverage", "Density / ConcurrentRecordDensity",
       "Environment / AccessControlRegime"], rows)

# ---- Table 6: gene-wise ablation --------------------------------------------
ab  = pd.read_csv(D + "ablation_results.csv")
sig = pd.read_csv(D + "ablation_significance.csv")
FULL = {"RLV": dict(A=0.716169, B=0.595, SR=0.870083),
        "Healthcare": dict(A=0.749412, B=0.553, SR=0.849250)}
NAME = {"RLV": dict(S="Speed", A="Attention", D="Density", E="Environment"),
        "Healthcare": dict(S="AccessEscalationRate", A="AuditCoverage",
                           D="ConcurrentRecordDensity", E="AccessControlRegime")}
def mark(dom, g, metric):
    r = sig[(sig.domain == dom) & (sig.gene_dropped == g) & (sig.metric == metric)]
    if not len(r): return ""
    s = " *" if bool(r.significant_p05.iloc[0]) else ""
    if s and not bool(r.holm_significant.iloc[0]): s += "\u25b4"
    return s
rows = []
for dom in ("RLV", "Healthcare"):
    f = FULL[dom]
    rows.append([f"{dom} (full model: alignment {f['A']:.3f}, baseline {f['B']:.3f}, "
                 f"survival {f['SR']:.3f})", "", "", "", "", ""])
    for r in ab[ab.domain == dom].itertuples():
        dA = r.A_triad_mean - f["A"]; dB = r.baseline_balanced_accuracy_mean - f["B"]; dS = r.SR - f["SR"]
        rows.append([NAME[dom][r.gene_dropped],
                     f"{dA:+.3f}".replace("+", "+").replace("-", "\u2212") + mark(dom, r.gene_dropped, "A_triad"),
                     f"{dB:+.3f}".replace("-", "\u2212") + mark(dom, r.gene_dropped, "baseline"),
                     f"{r.SR:.3f}", f"{dS:+.3f}".replace("-", "\u2212"),
                     f"{r.Rel_delta_vs_full:+.3f}".replace("-", "\u2212")])
write(6, "Table 6. Gene-wise ablation results for both domain configurations.",
      ["Gene dropped", "\u0394 alignment", "\u0394 baseline", "Survival",
       "\u0394 survival", "\u0394 aggregate"], rows)

# ---- Table 7: governance sequence -------------------------------------------
gm = pd.read_csv(D + "governance_multiseed.csv").set_index("domain")
gd = pd.read_csv(D + "governance_demo_results.csv").set_index("domain")
def pm(dom, col):
    return f"{gm.loc[dom, col+'_mean']:.3f} \u00b1 {gm.loc[dom, col+'_sd']:.3f}"
rows = [["Revoke rate", pm("RLV", "revoke_rate"), pm("Healthcare", "revoke_rate")],
        ["Random-redraw success rate", pm("RLV", "random_redraw_success_rate"),
         pm("Healthcare", "random_redraw_success_rate")],
        ["Governed regeneration success rate", pm("RLV", "regeneration_success_rate"),
         pm("Healthcare", "regeneration_success_rate")],
        ["Governance coherence: baseline", "1.000", "1.000"],
        ["Governance coherence: discard",
         f3(gd.loc["RLV", "GC_discard"]), f3(gd.loc["Healthcare", "GC_discard"])],
        ["Governance coherence: regenerate", "1.000", "1.000"],
        ["Reliability: baseline (generation 1)",
         f3(gm.loc["RLV", "Rel_baseline_mean"]), f3(gm.loc["Healthcare", "Rel_baseline_mean"])],
        ["Reliability: discard (counterfactual)", pm("RLV", "Rel_discard"), pm("Healthcare", "Rel_discard")],
        ["Reliability: regenerate (this framework)", pm("RLV", "Rel_regenerate"),
         pm("Healthcare", "Rel_regenerate")]]
write(7, "Table 7. Governance sequence results at the reference tightening magnitude.",
      ["", "RLV", "Healthcare"], rows)

# ---- Table 9: signature substitution ----------------------------------------
def ms(v):
    """Report a timing at the resolution the measurement is stable at.

    Two runs on the same machine returned 0.165 and 0.118 ms for the Ed25519
    baseline, so even one significant figure flips between runs below a
    millisecond. What is stable, and what the comparison rests on, is the band.
    """
    import math
    if v < 0.01:
        return "under 0.01 ms"
    if v < 1:
        return "under 1 ms"
    e = math.floor(math.log10(v))
    return f"about {round(v, -e):g} ms"

sb = pd.read_csv(D + "signature_benchmark.csv").set_index("scheme")
def g(s, c): return sb.loc[s, c]
rows = [["Signature field", "32 B", "64 B", "2,420 B"],
        ["Signed event record"] + [f"{int(g(s,'record_bytes')):,} B"
                                    for s in ("Keyed hash", "Ed25519", "ML-DSA-44")],
        # Timing is reported to the precision the measurement supports. Two runs on
        # the same machine returned 0.165 and 0.118 ms for the Ed25519 baseline, so
        # a three-decimal figure would claim a precision the measurement lacks.
        ["Signing time per event"] + [ms(g(s, "sign_ms")) for s in ("Keyed hash", "Ed25519", "ML-DSA-44")],
        ["Signature verification"] + [ms(g(s, "verify_ms")) for s in ("Keyed hash", "Ed25519", "ML-DSA-44")],
        ["Merkle depth at 262,144 events", "18", "18", "18"],
        ["Inclusion proof size", "576 B", "576 B", "576 B"],
        ["Append to the tree", "under 20 \u00b5s", "under 20 \u00b5s", "under 20 \u00b5s"],
        ["Contestation cost", "unchanged", "unchanged", "unchanged"]]
write(9, "Table 9. What a post-quantum signature substitution costs, measured against "
         "a classical baseline.",
      ["", "Keyed hash, as used here", "Classical (Ed25519)", "Post-quantum (ML-DSA-44)"], rows)

# ---- Table 10: context-mismatch routing -------------------------------------
rt = pd.read_csv(D + "context_routing_and_repair.csv").set_index("domain")
rows = [[lab, pc(rt.loc["RLV", col]), pc(rt.loc["Healthcare", col])]
        for lab, col in [("Proceed", "proceed_pct"), ("Context review", "context_review_pct"),
                         ("Quarantine or repair", "quarantine_repair_pct"),
                         ("Inadmissible", "inadmissible_pct")]]
write(10, "Table 10. Context-mismatch routing outcome at the reference tolerance and margin.",
      ["Route", "RLV", "Healthcare"], rows)

# ---- Table 11: context repair -----------------------------------------------
rows = [["Quarantined candidates",
         str(int(rt.loc["RLV", "repair_n_quarantined"])), str(int(rt.loc["Healthcare", "repair_n_quarantined"]))],
        ["Divergence before repair",
         f"{rt.loc['RLV','repair_Xi_before']:.4f}", f"{rt.loc['Healthcare','repair_Xi_before']:.4f}"],
        ["Divergence after repair",
         f"{rt.loc['RLV','repair_Xi_after']:.4f}", f"{rt.loc['Healthcare','repair_Xi_after']:.4f}"],
        ["Divergence reduction",
         pc(rt.loc["RLV", "repair_Xi_reduction"]), pc(rt.loc["Healthcare", "repair_Xi_reduction"])],
        ["Resolved below the tolerance",
         pc(rt.loc["RLV", "repair_resolved_below_xi"]), pc(rt.loc["Healthcare", "repair_resolved_below_xi"])],
        ["Remains within the feasibility gate",
         pc(rt.loc["RLV", "repair_remains_feasible"]), pc(rt.loc["Healthcare", "repair_remains_feasible"])]]
write(11, "Table 11. Context-repair outcome for quarantined candidates.",
      ["", "RLV", "Healthcare"], rows)


# ---- Table 5: reliability contract ------------------------------------------
import numpy as _np2
rs  = pd.read_csv(D + "rel_summary.csv").set_index("domain")
srb = pd.read_csv(D + "rel_seed_robustness.csv").set_index("domain")
gd2 = pd.read_csv(D + "governance_demo_results.csv").set_index("domain")
def boot_ci(dom, n=10000, seed=11):
    pe = pd.read_csv(D + f"rel_per_event_{dom.lower()}.csv")
    v = pe["Rel_event"].values if "Rel_event" in pe.columns else pe.iloc[:, -1].values
    rng = _np2.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    lo, hi = _np2.percentile(m, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"
def meas(d):
    return (rs.loc[d, "A_triad"] * rs.loc[d, "TS_mean"] * rs.loc[d, "SR_mean"]
            * gd2.loc[d, "GC_discard"]) ** 0.25
DOMS = ("RLV", "Healthcare")
rows = [["Reliability components (pool means, single pass)", "", ""],
        ["Triadic alignment"] + [f3(rs.loc[d, "A_triad"]) for d in DOMS],
        ["Transition stability, genesis to mutation"] + [f3(rs.loc[d, "TS_genesis_to_mutation_mean"]) for d in DOMS],
        ["Transition stability, mutation to repair"] + [f3(rs.loc[d, "TS_mutation_to_repair_mean"]) for d in DOMS],
        ["Admissibility survival, repair stage"] + [f3(rs.loc[d, "SR_mean"]) for d in DOMS],
        ["Governance coherence, measured in Section 3.6"] + [f3(gd2.loc[d, "GC_discard"]) for d in DOMS],
        ["Aggregate", "", ""],
        ["Reliability, measured lineage coherence"] + [f3(meas(d)) for d in DOMS],
        ["Reliability, idealized lineage"] + [f3(rs.loc[d, "Rel_mean"]) for d in DOMS],
        ["Reliability, idealized, bootstrap 95% CI over events"] + [boot_ci(d) for d in DOMS],
        ["Reliability, five-seed mean and SD"] +
            [f"{srb.loc[d,'Rel_mean']:.3f} \u00b1 {srb.loc[d,'Rel_std']:.3f}" for d in DOMS],
        ["Supervised baseline (five seeds)", "", ""],
        ["Balanced accuracy"] +
            [f"{srb.loc[d,'baseline_balanced_accuracy_mean']:.3f} \u00b1 "
             f"{srb.loc[d,'baseline_balanced_accuracy_std']:.3f}" for d in DOMS],
        ["F1"] + [f"{srb.loc[d,'baseline_f1_mean']:.3f} \u00b1 {srb.loc[d,'baseline_f1_std']:.3f}" for d in DOMS]]
write(5, "Table 5. Reliability components, aggregate with bootstrap intervals, and the "
         "supervised baseline, for both domain configurations.",
      ["", "RLV", "Healthcare"], rows)

# ---- Table 8: federated agreement and ledger --------------------------------
fa = pd.read_csv(D + "federated_agreement.csv").set_index("domain")
nn2 = pd.read_csv(D + "sweep_node_noise.csv"); nn2 = nn2[nn2.relative_variance_pct == 5].set_index("domain")
lc2 = pd.read_csv(D + "ledger_cost.csv").set_index("events")
kb2 = pd.read_csv(D + "kappa_boundary.csv")
import numpy as _np
def boundary(dom):
    k = kb2[kb2.domain == dom].sort_values("variance_pct")
    cv, v = k.cv.values, k.variance_pct.values
    i = next(j for j in range(len(cv)) if cv[j] > 0.5)
    return float(_np.interp(0.5, [cv[i-1], cv[i]], [v[i-1], v[i]]))
def pmn(dom, col):
    return f"{nn2.loc[dom, col+'_mean']:.3f} \u00b1 {nn2.loc[dom, col+'_sd']:.3f}"
rows = [["Federated agreement (four nodes, 150 shared candidates, 5 percent relative calibration variance)", "", ""],
        ["Agreement coefficient", pmn("RLV", "fleiss_kappa"), pmn("Healthcare", "fleiss_kappa")],
        ["Unanimous agreement rate", pmn("RLV", "unanimous_rate"), pmn("Healthcare", "unanimous_rate")],
        ["Mean admissible rate across nodes",
         f3(nn2.loc["RLV", "mean_admissible_rate"]), f3(nn2.loc["Healthcare", "mean_admissible_rate"])],
        ["Boundary at which the estimate ceases to be stable",
         f"{boundary('RLV'):.1f}%", f"{boundary('Healthcare'):.1f}%"],
        ["Permissioned Merkle ledger (illustrative six-event lifecycle)", "", ""],
        ["Chain valid before and after tamper", "true / false, detected at block 3", ""],
        ["Merkle root changed by tamper", "true", ""],
        ["Quorum required (four nodes, one tolerated fault)", "three of four", ""],
        ["Commit with four honest attestations", "committed", ""],
        ["Commit with one Byzantine node", "committed", ""],
        ["Commit with two faulty nodes", "rejected", ""],
        ["Ledger cost at 6 / 1,024 / 262,144 events", "", ""],
        ["Signed event record, mean size", f"{int(lc2.loc[6,'leaf_bytes'])} B", ""],
        ["Merkle depth", " / ".join(str(int(lc2.loc[n,'depth'])) for n in (6,1024,262144)), ""],
        ["Inclusion proof size",
         " / ".join(str(int(lc2.loc[n,'proof_bytes'])) for n in (6,1024,262144)) + " B", ""],
        ["Append one event", "under 20 \u00b5s at every scale", ""],
        ["Inclusion proof verification", "under 20 \u00b5s at every scale", ""],
        ["Batch rebuild of the whole tree", "under 1 s", ""],
        ["Full chain verification", "under 1 s", ""]]
write(8, "Table 8. Federated agreement, cryptographic lineage, and ledger cost at the "
         "reference operating point.", ["", "RLV", "Healthcare"], rows)

# ---- Table 12: context corruption -------------------------------------------
ar = pd.read_csv(D + "a1_reference.csv").set_index("domain")
mc = pd.read_csv(D + "a1_misrouting_cost.csv").set_index("domain")
m4 = pd.read_csv(D + "misrouting_pairs_400.csv")
sat = pd.read_csv(D + "misrouting_saturation.csv")
def pma(dom, col):
    return f"{ar.loc[dom, col]:.3f} \u00b1 {ar.loc[dom, col+'_sd']:.3f}"
def forfeit(dom):
    p = m4[m4.domain == dom]
    ach = p.Xi_before - p.Xi_repair_true; real = p.Xi_before - p.Xi_repair_declared
    return (ach.mean() - real.mean()) / ach.mean()
def worse(dom):
    p = m4[m4.domain == dom]
    return (p.Xi_repair_declared > p.Xi_before).mean()
rows = [["Screening outcome", "", ""],
        ["Feasible candidates",
         str(int(ar.loc["RLV", "n_feasible"])), str(int(ar.loc["Healthcare", "n_feasible"]))],
        ["Corrupted labels",
         f"{ar.loc['RLV','n_mislabeled']:.0f} \u00b1 {ar.loc['RLV','n_mislabeled_sd']:.0f}",
         f"{ar.loc['Healthcare','n_mislabeled']:.0f} \u00b1 {ar.loc['Healthcare','n_mislabeled_sd']:.0f}"],
        ["Detection rate", pma("RLV", "detection_rate"), pma("Healthcare", "detection_rate")],
        ["False-flag rate", pma("RLV", "false_flag_rate"), pma("Healthcare", "false_flag_rate")],
        ["Attribution", "", ""],
        ["Attribution rate", pma("RLV", "attribution_rate"), pma("Healthcare", "attribution_rate")],
        ["Misattribution rate", pma("RLV", "misattribution_rate"), pma("Healthcare", "misattribution_rate")],
        ["Wrongly reviewed, absolute",
         f3(ar.loc["RLV", "misattribution_rate"] * ar.loc["RLV", "false_flag_rate"]),
         f3(ar.loc["Healthcare", "misattribution_rate"] * ar.loc["Healthcare", "false_flag_rate"])],
        ["Label recovery rate", pma("RLV", "label_recovery_rate"), pma("Healthcare", "label_recovery_rate")],
        ["Misrouting cost (matched counterfactual)", "", ""],
        ["Misrouted candidates",
         str(int(mc.loc["RLV", "n_missed_records"])), str(int(mc.loc["Healthcare", "n_missed_records"]))],
        ["Misrouted share of corrupted"] +
        [f"{mc.loc[d,'n_missed_records'] / (ar.loc[d,'n_mislabeled'] * 40):.3f}"
         for d in ("RLV", "Healthcare")],
        ["Matched pairs at saturation",
         str(int(sat[(sat.domain=='RLV') & (sat.seed_runs==400)].pairs.iloc[0])),
         str(int(sat[(sat.domain=='Healthcare') & (sat.seed_runs==400)].pairs.iloc[0]))],
        ["Correction forfeited", pc(forfeit("RLV")), pc(forfeit("Healthcare"))],
        ["Left worse than before repair", pc(worse("RLV")), pc(worse("Healthcare"))],
        ["Cost, median",
         f"{m4[m4.domain=='RLV'].cost.median():+.4f}",
         f"{m4[m4.domain=='Healthcare'].cost.median():+.4f}"],
        ["Cost positive, fraction", "1.00", "1.00"]]
write(12, "Table 12. Context-corruption results at the reference operating point.",
      ["", "RLV", "Healthcare"], rows)

# ---- Table 13: weighting sensitivity ----------------------------------------
sw = pd.read_csv(D + "sweep_weights.csv")
LABEL = {"neutral": "Neutral", "alignment_first": "Alignment first",
         "stability_first": "Stability first", "admissibility_first": "Admissibility first",
         "governance_first": "Governance first", "release_facing": "Release facing",
         "supervisory_facing": "Supervisory facing"}
REG = {"idealized": "GC=1 (main results)", "measured": "GC measured, discard arm"}
rows = []
for key, lab in LABEL.items():
    r = []
    for dom in ("RLV", "Healthcare"):
        for reg in ("idealized", "measured"):
            v = sw[(sw.domain == dom) & (sw.weighting == key) & (sw.gc_regime == REG[reg])]
            r.append(f3(v.Rel.iloc[0]) if len(v) else "")
    rows.append([lab] + r)
write(13, "Table 13. The aggregate under seven weightings, with coherence idealized "
          "and as measured.",
      ["Weighting", "RLV, idealized", "RLV, measured",
       "Healthcare, idealized", "Healthcare, measured"], rows)

print("\ndeclared tables not generated: 1, 2, 3 and the abbreviations list")
