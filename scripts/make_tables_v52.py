"""
make_tables_v52.py -- write the numeric tables of the revised manuscript (v52)
and of Supplementary S4 as CSV files under data/tables_v52/.

make_tables.py writes the tables of the submitted version (v51 numbering) and is
unchanged. This script reads the same inputs plus data/multiseed/*.csv and writes:

  table_03.csv  Reliability components, cross-layer diagnostic rows, 50-draw rows, baseline
  table_04.csv  Gene-wise ablation (single file) -- as make_tables table_06
  table_05.csv  Governance sequence (single file + 50-draw columns)
  table_06.csv  Federated agreement and lineage anchoring (upper block of v51 Table 8)
  table_07.csv  Context-mismatch routing + context-repair outcome (v51 Tables 10 and 11 merged)
  table_08.csv  Context-corruption results (upper block of v51 Table 12)
  table_09.csv  Weighting sweep -- as make_tables table_13
  table_10.csv  Proxy-architecture sensitivity (Section 3.13)
  S4_1 ... S4_7b.csv   Supplementary S4 tables
  S2_3b.csv, S2_8.csv  lower blocks moved to the Supplement
Tables 1 and 2 of v52 are textual (mechanism matrix, design contract) and are not generated.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data") + os.sep; M = D + "multiseed" + os.sep
OUT = D + "tables_v52" + os.sep; os.makedirs(OUT, exist_ok=True)
T51 = D + "tables" + os.sep
def ci(m, lo, hi): return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
def r3(x): return f"{x:.3f}"
DOMS = ["RLV", "Healthcare"]

# ---- Table 3: components + diagnostic + 50-draw + baseline
rel = pd.read_csv(D + "rel_summary.csv").set_index("domain")
rs = pd.read_csv(M + "rel_multiseed_summary.csv").set_index("domain")
rm = pd.read_csv(M + "rel_multiseed.csv").set_index(["k", "domain"]); gv50 = pd.read_csv(M + "governance_multiseed50.csv").set_index(["k", "domain"])
rows = []
def row(label, a, b): rows.append({"Quantity": label, "RLV": a, "Healthcare": b})
# Rows, order and labels follow Table 3 of the manuscript.
gd = pd.read_csv(D + "governance_demo_results.csv").set_index("domain")
row("Reliability components (pool means, single pass)", "", "")
for lab, c in [("Triadic alignment", "A_triad"),
               ("Transition stability, genesis to mutation", "TS_genesis_to_mutation_mean"),
               ("Transition stability, mutation to repair", "TS_mutation_to_repair_mean"),
               ("Admissibility survival, repair stage", "SR_mean")]:
    row(lab, r3(rel.loc["RLV", c]), r3(rel.loc["Healthcare", c]))
row("Governance coherence, discard arm of Section 3.6 (one minus the single-pass revoke rate)", r3(gd.loc["RLV", "GC_discard"]), r3(gd.loc["Healthcare", "GC_discard"]))
row("Aggregate", "", "")
# Single-file aggregate rows on the component route (weighted geometric mean of the
# pool component means), the same route as the measured row and the 50-draw rows.
# The per-event mean (Rel_mean) is a different estimator; it is kept on its own
# labelled row because the per-event bootstrap interval of Table 3 belongs to it.
sf_idl = {d: (rel.loc[d, "A_triad"] * rel.loc[d, "TS_mean"] * rel.loc[d, "SR_mean"]) ** 0.25 for d in DOMS}
sf_meas = {d: sf_idl[d] * gd.loc[d, "GC_discard"] ** 0.25 for d in DOMS}
def boot_ci(dom, n=10000, seed=11):
    # percentile bootstrap over events of the per-event aggregate (as make_tables.py)
    v = pd.read_csv(D + f"rel_per_event_{dom.lower()}.csv")
    v = v["Rel_event"].values if "Rel_event" in v.columns else v.iloc[:, -1].values
    g = np.random.default_rng(seed)
    lo, hi = np.percentile([v[g.integers(0, len(v), len(v))].mean() for _ in range(n)], [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"
row("Reliability, measured governance coherence", r3(sf_meas["RLV"]), r3(sf_meas["Healthcare"]))
row("Reliability, idealized lineage", r3(sf_idl["RLV"]), r3(sf_idl["Healthcare"]))
row("Reliability, idealized, per-event route, mean [bootstrap 95% CI over events]",
    *[f'{rel.loc[d, "Rel_mean"]:.3f} {boot_ci(d)}' for d in DOMS])
# Both 50-draw aggregate rows are computed on the SAME route: the weighted
# geometric mean of the per-draw component means. Reporting the idealized row
# from the per-event mean (rel_multiseed_summary.Rel_mean_mean) and the measured
# row from the component route made the two incomparable, and understated the
# governance contribution by about 0.015; see the revision note in CHANGELOG.
rng = np.random.default_rng(1); meas = {}; idl = {}
for dom in DOMS:
    a = rm.xs(dom, level=1); g = gv50.xs(dom, level=1).loc[a.index]
    base = (a.A_triad * a.TS_mean * a.SR_mean) ** 0.25          # GC = 1
    relm = (a.A_triad * a.TS_mean * a.SR_mean * g.GC_discard) ** 0.25
    for store, arr in ((idl, base), (meas, relm)):
        b = [arr.values[rng.integers(0, len(arr), len(arr))].mean() for _ in range(10000)]
        store[dom] = ci(arr.mean(), *np.quantile(b, [0.025, 0.975]))
row("Reliability, idealized lineage, 50 generator draws, mean [95% CI]", idl["RLV"], idl["Healthcare"])
row("Reliability, measured governance coherence, 50 generator draws, mean [95% CI]", meas["RLV"], meas["Healthcare"])
# Cross-layer rows: the single generated file, then the same quantity over the
# fifty generator draws. The proxy was the one quantity the revision's fifty-draw
# treatment had not reached, and the single file sits at the favourable end of
# the distribution, so both are reported.
clx = pd.read_csv(D + "cross_layer_coherence.csv").set_index("domain")
clm = pd.read_csv(M + "cross_layer_multiseed_summary.csv").set_index("domain")
for c, lab in [("CL_lev", "Cross-layer level CL_lev, Equation (2.16); diagnostic, outside the aggregate"),
               ("CL_bal", "Cross-layer balance CL_bal, Equation (2.16)")]:
    row(lab, r3(clx.loc["RLV", c]), r3(clx.loc["Healthcare", c]))
    row(lab.split(",")[0] + ", 50 generator draws, mean [95% CI]",
        ci(clm.loc["RLV", c + "_mean"], clm.loc["RLV", c + "_ci_lo"], clm.loc["RLV", c + "_ci_hi"]),
        ci(clm.loc["Healthcare", c + "_mean"], clm.loc["Healthcare", c + "_ci_lo"], clm.loc["Healthcare", c + "_ci_hi"]))
row("Coherence flag Coh, Equation (2.17), tau_lev = 0.70, tau_bal = 0.80",
    *["coherent" if (clx.loc[d, "CL_lev"] >= 0.70 and clx.loc[d, "CL_bal"] >= 0.80) else "not coherent" for d in DOMS])
srb = pd.read_csv(D + "rel_seed_robustness.csv").set_index("domain")
row("Supervised baseline (five seeds)", "", "")
for lab, c in [("Balanced accuracy", "baseline_balanced_accuracy"), ("F1", "baseline_f1")]:
    row(lab, *[f'{srb.loc[d, c + "_mean"]:.3f} ± {srb.loc[d, c + "_std"]:.3f}' for d in DOMS])

if "CL_lev" in rel.columns:
    row("Coherence flag Coh (tau_lev 0.70, tau_bal 0.80)", *["coherent" if (rel.loc[d, "CL_lev"] >= 0.70 and rel.loc[d, "CL_bal"] >= 0.80) else "not coherent" for d in DOMS])
pd.DataFrame(rows).to_csv(OUT + "table_03.csv", index=False)

# ---- Tables 4 and 9: unchanged from v51 (table_06, table_13)
for src, dst in [("table_06.csv", "table_04.csv"), ("table_13.csv", "table_09.csv")]:
    if os.path.exists(T51 + src):
        df = pd.read_csv(T51 + src, skiprows=1); df = df.rename(columns={c: "Quantity" for c in df.columns if str(c).startswith("Unnamed")}); df.to_csv(OUT + dst, index=False)

# ---- Table 5: governance, single-file columns from v51 table_07 + 50-draw columns
g50 = pd.read_csv(M + "governance_multiseed50_summary.csv").set_index("domain")
t7 = pd.read_csv(T51 + "table_07.csv", skiprows=1) if os.path.exists(T51 + "table_07.csv") else pd.DataFrame()
add = pd.DataFrame([{"Quantity": q, "RLV (50 draws)": ci(g50.loc["RLV", c + "_mean"], g50.loc["RLV", c + "_ci_lo"], g50.loc["RLV", c + "_ci_hi"]), "Healthcare (50 draws)": ci(g50.loc["Healthcare", c + "_mean"], g50.loc["Healthcare", c + "_ci_lo"], g50.loc["Healthcare", c + "_ci_hi"])}
                    # GC_discard is carried in the 50-draw block as well: it is exactly 1 - revoke rate,
                    # so reporting the coherence from the single file next to a 50-draw revoke rate
                    # broke that identity across rows of the same table.
                    for q, c in [("Revoke rate", "revoke_rate"), ("Random-redraw success", "random_redraw_success_rate"), ("Governed regeneration success", "regeneration_success_rate"), ("Governance coherence: discard", "GC_discard"), ("Rel: baseline", "Rel_baseline"), ("Rel: discard", "Rel_discard"), ("Rel: regenerate", "Rel_regenerate"), ("Gain regenerate - discard", "Rel_gap_regen_minus_discard")]])
t7 = t7.rename(columns={t7.columns[0]: "Quantity", "RLV": "RLV (five seeds)", "Healthcare": "Healthcare (five seeds)"})
# The discard coherence is reported from the same five-seed run as the other rows of the
# table (1 - revoke rate of that run); the submitted table carried the single-pass value.
g5 = pd.read_csv(D + "governance_multiseed.csv").set_index("domain")
t7.loc[t7.Quantity == "Governance coherence: discard", ["RLV (five seeds)", "Healthcare (five seeds)"]] = [
    f'{g5.loc[d, "GC_discard_mean"]:.3f} ± {g5.loc[d, "GC_discard_sd"]:.3f}' for d in DOMS]
sep = pd.DataFrame([{"Quantity": "Fifty generator draws, mean [95% CI]"}])
pd.concat([t7, sep, add], ignore_index=True).to_csv(OUT + "table_05.csv", index=False)

# ---- Table 6 / S2.8: split v51 table_08 at the ledger-cost block
if os.path.exists(T51 + "table_08.csv"):
    t8 = pd.read_csv(T51 + "table_08.csv", skiprows=1); first = t8.iloc[:, 0].fillna("").astype(str)
    cut = next((i for i, v in enumerate(first) if v.lower().startswith("ledger")), len(t8))
    t8 = t8.rename(columns={t8.columns[0]: "Quantity"}); t8.iloc[:cut].to_csv(OUT + "table_06.csv", index=False); t8.iloc[cut:].to_csv(OUT + "S2_8.csv", index=False)
# ---- Table 7: merge v51 tables 10 and 11
if os.path.exists(T51 + "table_10.csv") and os.path.exists(T51 + "table_11.csv"):
    a = pd.read_csv(T51 + "table_10.csv", skiprows=1); b = pd.read_csv(T51 + "table_11.csv", skiprows=1); b.columns = a.columns[: len(b.columns)] if len(b.columns) <= len(a.columns) else b.columns
    sep = pd.DataFrame([["Context-repair outcome for quarantined candidates"] + [""] * (len(a.columns) - 1)], columns=a.columns)
    pd.concat([a, sep, b], ignore_index=True).to_csv(OUT + "table_07.csv", index=False)
# ---- Table 8 / S2.3b: split v51 table_12 at the misrouting block
if os.path.exists(T51 + "table_12.csv"):
    t12 = pd.read_csv(T51 + "table_12.csv", skiprows=1); first = t12.iloc[:, 0].fillna("").astype(str)
    cut = next((i for i, v in enumerate(first) if "misrout" in v.lower() or "matched" in v.lower()), len(t12))
    t12 = t12.rename(columns={t12.columns[0]: "Quantity"}); t12.iloc[:cut].to_csv(OUT + "table_08.csv", index=False); t12.iloc[cut:].to_csv(OUT + "S2_3b.csv", index=False)

# ---- Table 10: proxy architecture
pa = pd.read_csv(M + "proxy_arch_sweep_summary.csv"); arms = ["mlp1_tanh_s018_r2", "mlp1_relu", "mlp2_tanh", "mlp1_s009", "mlp1_s036", "mlp1_r1", "mlp1_r4", "attn1"]
rows = []
for arm in arms:
    r = {d: pa[(pa.arm == arm) & (pa.domain == d)].iloc[0] for d in DOMS}
    rows.append({"Proxy arm": arm, "dRel RLV": f"{r['RLV'].dRel_mean_mean:+.3f} [{r['RLV'].dRel_mean_ci_lo:+.3f}, {r['RLV'].dRel_mean_ci_hi:+.3f}]", "dRel Healthcare": f"{r['Healthcare'].dRel_mean_mean:+.3f} [{r['Healthcare'].dRel_mean_ci_lo:+.3f}, {r['Healthcare'].dRel_mean_ci_hi:+.3f}]",
                 "dSR (RLV / HC)": f"{r['RLV'].dSR_mean_mean:+.3f} / {r['Healthcare'].dSR_mean_mean:+.3f}", "Gate agreement (RLV / HC)": f"{r['RLV'].Ag_agreement_vs_ref_mean:.3f} / {r['Healthcare'].Ag_agreement_vs_ref_mean:.3f}"})
pd.DataFrame(rows).to_csv(OUT + "table_10.csv", index=False)

# ---- Supplementary S4 tables (straight copies of the summary CSVs, full precision)
for src, dst in [("rel_multiseed_summary.csv", "S4_1.csv"), ("ablation_multiseed_summary.csv", "S4_2.csv"), ("governance_multiseed50_summary.csv", "S4_3.csv"), ("sweep_node_noise50.csv", "S4_4.csv"),
                 ("sr_floor_eps_sweep.csv", "S4_5a.csv"), ("sr_floor_delta_sweep.csv", "S4_5b.csv"), ("xi_evaluation.csv", "S4_6a.csv"), ("xi_evaluation_band.csv", "S4_6b.csv"),
                 ("attribution_matched_summary.csv", "S4_7a.csv"), ("attribution_width_sweep.csv", "S4_7b.csv")]:
    df = pd.read_csv(M + src)
    if dst == "S4_7b.csv": df = df.groupby(["domain", "width_x_halfwidth"]).mean(numeric_only=True).reset_index()
    df.to_csv(OUT + dst, index=False)  # full precision; the tables print rounded values
print("v52 tables written to", OUT, ":", sorted(os.listdir(OUT)))
