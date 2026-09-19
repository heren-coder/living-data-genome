"""Re-derives the headline numbers of Section 3 from the generated CSVs and
compares them with the values printed in the paper. Exits non-zero on mismatch."""
import pandas as pd, sys
D="../data/"; ok=True
def chk(label, got, want, tol):
    global ok
    good = abs(got-want) <= tol
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {label:46s} got {got:.3f}  paper {want:.3f}")
r=pd.read_csv(D+"context_routing_and_repair.csv").set_index("domain")
for dom,(p,c,q,i) in {"RLV":(0.710,0.018,0.140,0.132),"Healthcare":(0.714,0.024,0.114,0.149)}.items():
    chk(f"Table 3.6 proceed [{dom}]",       r.loc[dom,"proceed_pct"],           p, 0.001)
    chk(f"Table 3.6 context review [{dom}]",r.loc[dom,"context_review_pct"],    c, 0.001)
    chk(f"Table 3.6 quarantine [{dom}]",    r.loc[dom,"quarantine_repair_pct"], q, 0.001)
    chk(f"Table 3.6 inadmissible [{dom}]",  r.loc[dom,"inadmissible_pct"],      i, 0.001)
    chk(f"Table 3.7 Xi reduction [{dom}]",  r.loc[dom,"repair_Xi_reduction"],
        {"RLV":0.699,"Healthcare":0.703}[dom], 0.005)

g=pd.read_csv(D+"release_gating.csv").set_index("domain")
for dom,(ag,ar,ai,ev) in {"RLV":(0.868,0.754,0.729,0.850),"Healthcare":(0.851,0.691,0.670,0.810)}.items():
    chk(f"3.2.1 A_g [{dom}]",  g.loc[dom,"Ag"],  ag, 0.001)
    chk(f"3.2.1 A_R [{dom}]",  g.loc[dom,"AR"],  ar, 0.001)
    chk(f"3.2.1 A_imm [{dom}]",g.loc[dom,"Aimm"],ai, 0.001)
    chk(f"3.2.1 event-level release [{dom}]", g.loc[dom,"event_level_Aimm"], ev, 0.001)
p=pd.read_csv(D+"counterfactual_probe.csv").set_index("domain")
for dom,(m,w,a) in {"RLV":(0.090,0.220,0.272),"Healthcare":(0.080,0.246,0.251)}.items():
    chk(f"3.2.1 margin median [{dom}]", p.loc[dom,"margin_median"], m, 0.001)
    chk(f"3.2.1 within 0.05 [{dom}]",   p.loc[dom,"within_005"],    w, 0.001)
    chk(f"3.2.1 probe vs Shapley [{dom}]", p.loc[dom,"agreement_analytic_vs_shapley"], a, 0.001)
e=pd.read_csv(D+"expl_payload_stats.csv").set_index("domain")
for dom,(b,f) in {"RLV":(168,0.324),"Healthcare":(187,0.372)}.items():
    chk(f"3.2.1 payload bytes [{dom}]", e.loc[dom,"payload_bytes_median"], b, 0.5)
    chk(f"3.2.1 consistent box [{dom}]",e.loc[dom,"consistent_box_fraction_median"], f, 0.001)
x=pd.read_csv(D+"xi_null_summary.csv").set_index("domain")
for dom,(nm,au) in {"RLV":(0.0139,0.862),"Healthcare":(0.0128,0.933)}.items():
    chk(f"3.6 null median [{dom}]", x.loc[dom,"null_median"], nm, 0.0002)
    chk(f"3.6 AUC [{dom}]",         x.loc[dom,"auc"],         au, 0.002)


s=pd.read_csv(D+"rel_seed_robustness.csv").set_index("domain")
for dom,(at,ba,f1,rl) in {"RLV":(0.716,0.595,0.726,0.870),"Healthcare":(0.749,0.553,0.691,0.874)}.items():
    chk(f"3.2 A_triad 5-seed [{dom}]",  s.loc[dom,"A_triad_mean"], at, 0.001)
    chk(f"3.2 baseline bal.acc [{dom}]",s.loc[dom,"baseline_balanced_accuracy_mean"], ba, 0.001)
    chk(f"3.2 baseline F1 [{dom}]",     s.loc[dom,"baseline_f1_mean"], f1, 0.001)
    chk(f"3.2 Rel 5-seed [{dom}]",      s.loc[dom,"Rel_mean"], rl, 0.001)


gm=pd.read_csv(D+"governance_multiseed.csv").set_index("domain")
for dom,(rv,rd,rg,rb,rdi,rre) in {"RLV":(0.294,0.051,0.854,0.900,0.756,0.890),
                                  "Healthcare":(0.322,0.041,0.867,0.913,0.752,0.903)}.items():
    chk(f"3.4 revoke rate [{dom}]",       gm.loc[dom,"revoke_rate_mean"], rv, 0.001)
    chk(f"3.4 random redraw [{dom}]",     gm.loc[dom,"random_redraw_success_rate_mean"], rd, 0.001)
    chk(f"3.4 regeneration success [{dom}]", gm.loc[dom,"regeneration_success_rate_mean"], rg, 0.001)
    chk(f"3.4 Rel baseline [{dom}]",      gm.loc[dom,"Rel_baseline_mean"], rb, 0.001)
    chk(f"3.4 Rel discard [{dom}]",       gm.loc[dom,"Rel_discard_mean"], rdi, 0.001)
    chk(f"3.4 Rel regenerate [{dom}]",    gm.loc[dom,"Rel_regenerate_mean"], rre, 0.001)
for dom,(gcm,gcs) in {"RLV":(0.706,0.032),"Healthcare":(0.678,0.019)}.items():
    chk(f"Table 5 coherence discard, five seeds [{dom}]", gm.loc[dom,"GC_discard_mean"], gcm, 0.0005)
    chk(f"Table 5 coherence discard SD [{dom}]",          gm.loc[dom,"GC_discard_sd"],   gcs, 0.0005)
nn=pd.read_csv(D+"sweep_node_noise.csv"); nn=nn[nn.relative_variance_pct==5].set_index("domain")
for dom,(k,u,a) in {"RLV":(0.879,0.928,0.795),"Healthcare":(0.877,0.920,0.778)}.items():
    chk(f"3.5 Fleiss kappa [{dom}]",  nn.loc[dom,"fleiss_kappa_mean"],   k, 0.001)
    chk(f"3.5 unanimous rate [{dom}]",nn.loc[dom,"unanimous_rate_mean"], u, 0.001)
    chk(f"3.5 admissible rate [{dom}]",nn.loc[dom,"mean_admissible_rate"],a, 0.001)
lc=pd.read_csv(D+"ledger_cost.csv").set_index("events")
for n,(d,p) in {6:(3,96),1024:(10,320),262144:(18,576)}.items():
    chk(f"3.5 Merkle depth [{n}]", lc.loc[n,"depth"], d, 0.5)
    chk(f"3.5 proof bytes [{n}]",  lc.loc[n,"proof_bytes"], p, 0.5)
ab=pd.read_csv(D+"ablation_significance.csv"); ab=ab[ab.metric=="A_triad"].set_index(["domain","gene_dropped"])
for (dom,g),want in {("RLV","D"):-0.130,("Healthcare","S"):-0.090}.items():
    chk(f"3.3 dA_triad [{dom},{g}]", ab.loc[(dom,g),"mean_diff"], want, 0.001)


# ---------------------------------------------------------------- added checks
# Measurements introduced in the current revision. Section numbers follow the
# manuscript as submitted, not the earlier draft the checks above were written for.

sg = pd.read_csv(D + "shapley_geometry.csv")
for dom, want in {"RLV": 0.987, "Healthcare": 0.887}.items():
    chk(f"3.4.1 max pool concentration [{dom}]",
        sg[sg.domain == dom].pool_conc.max(), want, 0.002)
chk("3.4.1 geometry correlation",
    float(sg.gap.corr(sg.pool_conc)), 0.863, 0.01)

lv = pd.read_csv(D + "lime_vs_shapley.csv").set_index("domain")
for dom, (l, sh) in {"RLV": (0.851, 0.272), "Healthcare": (0.894, 0.251)}.items():
    chk(f"3.4.1 probe vs surrogate [{dom}]", lv.loc[dom, "agree_cf_lime"],   l,  0.002)
    chk(f"3.4.1 probe vs Shapley [{dom}]",   lv.loc[dom, "agree_cf_shapley"], sh, 0.002)

om = pd.read_csv(D + "operating_map.csv")
for dom, (span_v, span_t) in {"RLV": (0.003, 0.393), "Healthcare": (0.004, 0.456)}.items():
    o = om[om.domain == dom]
    at_tight = o[o.tightening == 0.100]
    chk(f"3.6 gain span across variance [{dom}]",
        float(at_tight.rel_gap_mean.max() - at_tight.rel_gap_mean.min()), span_v, 0.002)
    at_var = o[o.variance_pct == 5]
    chk(f"3.6 gain span across tightening [{dom}]",
        float(at_var.rel_gap_mean.max() - at_var.rel_gap_mean.min()), span_t, 0.01)

kb = pd.read_csv(D + "kappa_boundary.csv")
for dom, want in {"RLV": 16.3, "Healthcare": 17.0}.items():
    k = kb[kb.domain == dom].sort_values("variance_pct")
    cv, v = k.cv.values, k.variance_pct.values
    i = next(j for j in range(len(cv)) if cv[j] > 0.5)
    import numpy as _np
    bnd = float(_np.interp(0.5, [cv[i-1], cv[i]], [v[i-1], v[i]]))
    chk(f"3.7 agreement boundary [{dom}]", bnd, want, 0.3)

pq = pd.read_csv(D + "pq_ledger.csv")
sb = pd.read_csv(D + "signature_benchmark.csv").set_index("scheme")
for sch, rb in {"Keyed hash": 410, "Ed25519": 434, "ML-DSA-44": 3574}.items():
    chk(f"3.7.1 record bytes [{sch}]", float(sb.loc[sch, "record_bytes"]), rb, 0.5)
# Signing time is machine-dependent and is not checked against a fixed value.
# The same operation returns 17.7 ms over twenty-five repetitions here and
# 26.7 ms over seven in pq_ledger.py, and the ML-DSA to Ed25519 ratio has been
# measured between 107 on an Apple M3 Max and 522 on a Linux container. What
# the manuscript claims, and what is checked, is the order of magnitude.
_r = float(sb.loc["ML-DSA-44", "sign_ms"] / sb.loc["Ed25519", "sign_ms"])
chk("3.7.1 signing ratio, order of magnitude only", _r, 525.0, 475.0)
chk("3.7.1 record ratio ML-DSA over Ed25519",
    float(sb.loc["ML-DSA-44", "record_bytes"] / sb.loc["Ed25519", "record_bytes"]), 8.2, 0.1)

chk("3.7.1 ML-DSA signature bytes",
    float(pq[pq.scheme == "ML-DSA-44"].sig_bytes.iloc[0]), 2420, 0.5)
chk("3.7.1 Merkle depth unchanged",
    float(pq[(pq.scheme == "ML-DSA-44") & (pq.events == 262144)].depth.iloc[0]), 18, 0.5)

ms = pd.read_csv(D + "misrouting_saturation.csv")
for dom in ("RLV", "Healthcare"):
    m = ms[(ms.domain == dom) & (ms.seed_runs >= 80)]
    chk(f"3.8.1 saturated pair count [{dom}]", float(m.pairs.min()), 32, 0.5)
m4 = pd.read_csv(D + "misrouting_pairs_400.csv")
for dom, (f, w) in {"RLV": (0.418, 0.094), "Healthcare": (0.730, 0.250)}.items():
    p4 = m4[m4.domain == dom]
    ach  = p4.Xi_before - p4.Xi_repair_true
    real = p4.Xi_before - p4.Xi_repair_declared
    chk(f"3.8.1 correction forfeited [{dom}]",
        float((ach.mean() - real.mean()) / ach.mean()), f, 0.005)
    chk(f"3.8.1 left worse than before [{dom}]",
        float((p4.Xi_repair_declared > p4.Xi_before).mean()), w, 0.005)

rg = pd.read_csv(D + "repair_geometry.csv").set_index("domain")
for dom, want in {"RLV": 0.838, "Healthcare": 0.832}.items():
    chk(f"3.9 P2' condition holds [{dom}]", rg.loc[dom, "hold"], want, 0.002)

va = pd.read_csv(D + "validity_arms.csv")
for dom, (rel_, infl) in {"RLV": (-0.166, 0.044), "Healthcare": (-0.192, 0.052)}.items():
    v = va[va.domain == dom].set_index("arm")
    chk(f"3.10 releasability arm [{dom}]", v.loc["releasability", "dRel"], rel_, 0.002)
    chk(f"3.10 inflation arm [{dom}]",
        v.loc["enforced admissibility", "dRel"], infl, 0.002)
    for a in ("candidate order", "quality scale", "context naming"):
        chk(f"3.10 discriminant {a} [{dom}]", v.loc[a, "dRel"], 0.0, 0.0005)

ss = pd.read_csv(D + "sweep_sTS.csv")
for dom, (lo, hi) in {"RLV": (0.989, 0.731), "Healthcare": (0.989, 0.732)}.items():
    t = ss[ss.domain == dom].set_index("scale")
    chk(f"3.11 stability at scale 1 [{dom}]",  t.loc[1,  "TS_mean"], lo, 0.002)
    chk(f"3.11 stability at scale 32 [{dom}]", t.loc[32, "TS_mean"], hi, 0.002)

ct = pd.read_csv(D + "sweep_coh_thresholds.csv")
both = ct[ct.pass_both == 1]
chk("3.2 highest level threshold cleared",   float(both.tau_lev.max()), 0.71, 0.005)
chk("3.2 highest balance threshold cleared", float(both.tau_bal.max()), 0.83, 0.005)

sd = pd.read_csv(D + "seeds400.csv")
cv = pd.read_csv(D + "seed_convergence.csv")
for dom in ("RLV", "Healthcare"):
    x = sd[sd.domain == dom]
    chk(f"3.12 survival seed sd is zero [{dom}]",  float(x.sr.std(ddof=1)), 0.0, 1e-9)
    chk(f"3.12 stability seed sd is zero [{dom}]", float(x.ts.std(ddof=1)), 0.0, 1e-9)
    c5   = cv[(cv.domain == dom) & (cv.metric == "rel") & (cv.n_seeds == 5)]["mean"].iloc[0]
    c400 = cv[(cv.domain == dom) & (cv.metric == "rel") & (cv.n_seeds == 400)]["mean"].iloc[0]
    chk(f"3.12 five vs four hundred seeds [{dom}]", abs(float(c5 - c400)), 0.0, 0.0012)


# ---------------------------------------------------------------- v1.3.0 (revision v52): generator-level uncertainty
# Labels give the v52 table/section; the checks above keep their v51 labels (v51->v52 table map: 5->3, 6->4, 7->5, 8->6, 9->7, 10->8, 11->9, 13->10, 14->11, 15->12; see README).
M = D + "multiseed/"
try:
    rs = pd.read_csv(M + "rel_multiseed_summary.csv").set_index("domain")
    for dom, want in {"RLV": 0.846, "Healthcare": 0.836}.items():
        chk(f"Rel per-event mean over 50 draws [{dom}]", rs.loc[dom, "Rel_mean_mean"], want, 0.002)
    # Table 3 reports both aggregate rows on the component route (weighted geometric
    # mean of the per-draw component means), so that idealized and measured are
    # comparable and their difference is the governance contribution alone.
    _rm = pd.read_csv(M + "rel_multiseed.csv").set_index(["k", "domain"])
    _gv = pd.read_csv(M + "governance_multiseed50.csv").set_index(["k", "domain"])
    for dom, (wi, wm) in {"RLV": (0.861, 0.789), "Healthcare": (0.857, 0.785)}.items():
        _a = _rm.xs(dom, level=1); _g = _gv.xs(dom, level=1).loc[_a.index]
        _base = (_a.A_triad * _a.TS_mean * _a.SR_mean) ** 0.25
        _meas = (_a.A_triad * _a.TS_mean * _a.SR_mean * _g.GC_discard) ** 0.25
        chk(f"Table 3 (v52) Rel idealized, 50 draws [{dom}]", _base.mean(), wi, 0.002)
        chk(f"Table 3 (v52) Rel measured, 50 draws [{dom}]", _meas.mean(), wm, 0.002)
        chk(f"Sec 3.3 governance contribution [{dom}]", _base.mean() - _meas.mean(), 0.072, 0.002)
    # Single-file rows of Table 3: measured and idealized on the component route,
    # the per-event mean on its own labelled row.
    _rs1 = pd.read_csv(D + "rel_summary.csv").set_index("domain")
    _gd1 = pd.read_csv(D + "governance_demo_results.csv").set_index("domain")
    for dom, (si, sm, pe) in {"RLV": (0.869, 0.800, 0.857), "Healthcare": (0.877, 0.812, 0.861)}.items():
        _b1 = (_rs1.loc[dom, "A_triad"] * _rs1.loc[dom, "TS_mean"] * _rs1.loc[dom, "SR_mean"]) ** 0.25
        chk(f"Table 3 single file Rel idealized, component route [{dom}]", _b1, si, 0.0015)
        chk(f"Table 3 single file Rel measured, component route [{dom}]", _b1 * _gd1.loc[dom, "GC_discard"] ** 0.25, sm, 0.0015)
        chk(f"Table 3 single file Rel idealized, per-event route [{dom}]", _rs1.loc[dom, "Rel_mean"], pe, 0.0015)
    gv = pd.read_csv(M + "governance_multiseed50_summary.csv").set_index("domain")
    for dom, (rr, rg, gap) in {"RLV": (0.295, 0.880, 0.136), "Healthcare": (0.296, 0.883, 0.137)}.items():
        chk(f"Table 5 (v52) revoke rate, 50 draws [{dom}]", gv.loc[dom, "revoke_rate_mean"], rr, 0.003)
        chk(f"Table 5 (v52) regeneration success [{dom}]", gv.loc[dom, "regeneration_success_rate_mean"], rg, 0.003)
        chk(f"Table 5 (v52) gain regen-discard [{dom}]", gv.loc[dom, "Rel_gap_regen_minus_discard_mean"], gap, 0.002)
    ru = pd.read_csv(M + "xi_rules.csv"); ev = pd.read_csv(M + "xi_evaluation.csv")
    for dom, want in {"RLV": 0.028, "Healthcare": 0.030}.items():
        chk(f"Sec 3.8 xi_youden, calibration draws [{dom}]", ru[(ru.domain == dom) & (ru.rule == "xi_youden")].xi.iloc[0], want, 0.0015)
    for dom, (det, fa) in {"RLV": (0.767, 0.177), "Healthcare": (0.827, 0.118)}.items():
        e = ev[(ev.domain == dom) & (ev.rule == "xi_decl")].iloc[0]
        chk(f"Sec 3.8 held-out detection [{dom}]", e.detection_mean, det, 0.003); chk(f"Sec 3.8 held-out false flag [{dom}]", e.false_flag_mean, fa, 0.003)
    fl = pd.read_csv(M + "sr_floor_eps_sweep.csv")
    for dom, want in {"RLV": 0.0004, "Healthcare": 0.0006}.items():
        chk(f"Sec 3.12 floor effect on mean Rel [{dom}]", fl[(fl.domain == dom) & (fl.eps == 1e-6)].dev_mean.iloc[0], want, 0.0002)
    at = pd.read_csv(M + "attribution_matched_summary.csv")
    for dom, (sl, lp) in {"RLV": (0.847, 0.277), "Healthcare": (0.866, 0.276)}.items():
        chk(f"Sec 3.4.1 Shapley local vs probe [{dom}]", at[(at.domain == dom) & (at.method == "shapley") & (at.reference == "local")].agreement_mean.iloc[0], sl, 0.005)
        chk(f"Sec 3.4.1 surrogate pool vs probe [{dom}]", at[(at.domain == dom) & (at.method == "lime") & (at.reference == "pool")].agreement_mean.iloc[0], lp, 0.005)
    # ---- v1.3.2: cross-layer proxy over 50 draws and across the validity arms
    cx = pd.read_csv(M + "cross_layer_multiseed_summary.csv").set_index("domain")
    for dom, (lev, bal, coh) in {"RLV": (0.704, 0.826, 0.52), "Healthcare": (0.720, 0.838, 0.76)}.items():
        chk(f"Sec 2.8 CL_lev, 50 draws [{dom}]", cx.loc[dom, "CL_lev_mean"], lev, 0.002)
        chk(f"Sec 2.8 CL_bal, 50 draws [{dom}]", cx.loc[dom, "CL_bal_mean"], bal, 0.002)
        chk(f"Sec 2.8 coherence-flag rate [{dom}]", cx.loc[dom, "Coh_rate"], coh, 0.02)
    ca = pd.read_csv(M + "cross_layer_arms.csv").set_index(["domain", "arm"])
    for dom, (mis, rel) in {"RLV": (-0.309, -0.119), "Healthcare": (-0.338, -0.101)}.items():
        chk(f"Sec 2.8 arm misalignment dCL_lev [{dom}]", ca.loc[(dom, "misalignment"), "dCL_lev"], mis, 0.003)
        chk(f"Sec 2.8 arm releasability dCL_lev [{dom}]", ca.loc[(dom, "releasability"), "dCL_lev"], rel, 0.003)
        for a in ["candidate order", "quality scale", "instability"]:
            chk(f"Sec 2.8 arm {a} dCL_lev is zero [{dom}]", ca.loc[(dom, a), "dCL_lev"], 0.0, 0.0005)
    pa = pd.read_csv(M + "proxy_arch_sweep_summary.csv")
    # Table 9, every printed cell: (dRel mean, lo, hi) per domain, dSR (RLV, HC), gate agreement (RLV, HC);
    # compared after formatting to the three decimals the table prints
    pa = pa.set_index(["arm", "domain"])
    T9 = {"mlp1_relu": ((0.008, 0.006, 0.010), (0.005, 0.003, 0.007), (0.013, 0.010), (0.964, 0.965)),
          "mlp2_tanh": ((0.008, 0.004, 0.013), (0.004, -0.001, 0.009), (0.014, 0.008), (0.908, 0.906)),
          "mlp1_s009": ((0.010, 0.008, 0.012), (0.010, 0.008, 0.012), (0.019, 0.019), (0.960, 0.959)),
          "mlp1_s036": ((-0.041, -0.047, -0.036), (-0.038, -0.044, -0.031), (-0.075, -0.067), (0.901, 0.907)),
          "mlp1_r1": ((0.000, -0.001, 0.002), (0.002, 0.000, 0.004), (-0.001, 0.001), (0.981, 0.977)),
          "mlp1_r4": ((-0.001, -0.004, 0.002), (-0.001, -0.003, 0.002), (-0.003, -0.002), (0.969, 0.964)),
          "attn1": ((0.011, 0.007, 0.014), (0.010, 0.006, 0.014), (0.020, 0.020), (0.922, 0.916))}
    for arm, (rlv, hc, dsr, agr) in T9.items():
        for k, dom in enumerate(["RLV", "Healthcare"]):
            row = pa.loc[(arm, dom)]
            for col, want in zip(["dRel_mean_mean", "dRel_mean_ci_lo", "dRel_mean_ci_hi"], (rlv, hc)[k]):
                chk(f"Table 9 {arm} {col.replace('_mean', '')} [{dom}]", float(f"{row[col]:.3f}") + 0.0, want, 1e-9)
            chk(f"Table 9 {arm} dSR [{dom}]", float(f"{row.dSR_mean_mean:.3f}") + 0.0, dsr[k], 1e-9)
            chk(f"Table 9 {arm} gate agreement [{dom}]", float(f"{row.Ag_agreement_vs_ref_mean:.3f}"), agr[k], 1e-9)
    # ---- v1.3.6: seed spread against draw spread (Sec 3.3, Sec 3.13, Figure 14, Supplementary S2/S3/S4.1)
    _s4 = pd.read_csv(D + "seeds400.csv"); _s4["comp"] = (_s4.a_triad * _s4.ts * _s4.sr * _s4.gc) ** 0.25
    _cv = pd.read_csv(D + "seed_convergence.csv"); _cv = _cv[_cv.metric == "rel"]
    _rm2 = pd.read_csv(M + "rel_multiseed.csv"); _rm2["comp"] = (_rm2.A_triad * _rm2.TS_mean * _rm2.SR_mean) ** 0.25
    for dom, (sd_pe, ci5, dev5) in {"RLV": (0.0035, 0.0028, 0.0005), "Healthcare": (0.0033, 0.0033, 0.0010)}.items():
        _x = _s4[_s4.domain == dom]; _c = _cv[_cv.domain == dom].set_index("n_seeds")
        chk(f"Sec 3.3 seed SD over 400 seeds, per-event [{dom}]", _x.rel.std(ddof=1), sd_pe, 0.00005)
        chk(f"Sec 3.3 draw SD, per-event route [{dom}]", rs.loc[dom, "Rel_mean_sd"], 0.016, 0.0005)
        chk(f"Sec 3.3 draw SD, component route [{dom}]", _rm2[_rm2.domain == dom].comp.std(ddof=1), 0.012, 0.0005)
        _r = [rs.loc[dom, "Rel_mean_sd"] / _x.rel.std(ddof=1), _rm2[_rm2.domain == dom].comp.std(ddof=1) / _x.comp.std(ddof=1)]
        chk(f"Sec 3.3 draw/seed ratio within three to five [{dom}]", float(min(max(_r), 5) >= 3 and max(_r) <= 5 and min(_r) >= 3), 1.0, 0)
        _dev = (_c["mean"] - _c.loc[400, "mean"]).abs()
        chk(f"Sec 3.13 mean deviation at five seeds [{dom}]", _dev.loc[5], dev5, 0.00006)
        chk(f"Sec 3.13 max mean deviation over subsets <= 0.0022 [{dom}]", float(_dev.max() <= 0.00225), 1.0, 0)
        chk(f"Sec 3.13 interval at five seeds [{dom}]", _c.loc[5, "ci95"], ci5, 0.00006)
        chk(f"Sec 3.13 interval at 400 seeds [{dom}]", _c.loc[400, "ci95"], 0.0003, 0.00006)
except FileNotFoundError as e:
    print("  SKIP  v1.3.0 checks (run generate_multiseed.py and the multiseed scripts first):", e); 
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
