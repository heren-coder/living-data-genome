# The Living Data Genome — reproducibility package

Code and data for *The Living Data Genome: Ledger-Governed Evidence Lifecycles for
Accountable AI under Limited Observability* (H. Eren, submitted to *Systems*, MDPI).
Version 1.3.7 is the version reported in the revised manuscript. It is archived on
Zenodo under the concept DOI [10.5281/zenodo.21862909](https://doi.org/10.5281/zenodo.21862909),
which always resolves to the latest version and lists the version DOIs.

Every number the paper measures is written by a script in this package, and a final
verification step compares the written files with the values printed in the
manuscript. The scenarios are synthetic: a seeded simulator produces the two
configurations of the paper (post-incident red-light violation reasoning and a
healthcare access-governance vignette), and everything downstream is computed from
its output.

## Running

    pip install -r requirements.txt
    cd scripts
    bash run_all.sh

`requirements.txt` pins the environment that produced the published numbers
(Python 3.12.13, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0,
matplotlib 3.10.8, pillow 12.3.0, dilithium-py 1.4.0, cryptography 50.0.0), and each
run records its own environment in `data/env.json` (`record_env.py`).

`run_all.sh` is the single entry point. It resolves paths from its own location, so
the package can be unpacked anywhere, and runs the whole sequence in dependency
order: generation, the reliability protocol, ablations, governance, federation and
ledger, context screening, release gate and attribution, protocol sensitivity, the
uncertainty analysis over fifty generator draws, tables, figures, and finally
`verify_paper_numbers.py`. The verification performs 272 checks against the values
printed in the manuscript and exits non-zero if any fails; a run that ends with
`ALL CHECKS PASSED` has reproduced the paper. One check, the post-quantum signing
ratio of Supplementary S7.1, is an order of magnitude rather than a value, because
signing time depends on the machine (the ratio has been measured between 107 and 522);
Table S2.7 reports the timings as bands for the same reason.

A full run from a fresh clone took 27 minutes on an Apple M3 Max, most of it in the
fifty-draw group, the misrouting saturation sweep and the joint parameter sweep
`operating_map.py`. Apart from the three files that record timings (`ledger_cost.csv`,
`pq_ledger.csv`, `signature_benchmark.csv`), every data file it writes matches the
released one to within floating-point rounding (below 1e-12), and the regenerated
figures match the released files up to the anti-aliasing of individual pixels.

## Seeds

The pipeline is deterministic under the seeds declared in Supplementary S3. The
reference scenario file uses generator seed 42 (red-light violation) and 43
(healthcare); the fifty draws use 42 + 1000k and 42 + 1000k + 1, k = 0, …, 49, and the
first draw reproduces `data/scenarios.csv` byte for byte. Evaluation seeds and the
fixed offsets used by auxiliary random streams (bootstrap resampling, attribution
backgrounds, perturbation arms, sweep repetitions) are stated in each script. The
per-draw scenario and trace files are not tracked (`.gitignore`); `generate_multiseed.py`
writes them at the start of the fifty-draw group.

## What is computed and what is declared

- **Computed.** Every file under `data/` is written by a script in `scripts/`, and every
  numeric table is written as CSV by `make_tables_v52.py` (revised manuscript, under
  `data/tables_v52/`) or `make_tables.py` (submitted version, under `data/tables/`).
- **Declared.** Tables 1 and 2 and Supplementary Tables S3.1–S3.4 state the mechanism
  comparison, the protocol constants, the layer assignment, the vocabulary contract
  and the dominance profiles. They are declarations rather than measurements and are
  not generated; the constants they list are the ones used in the scripts.
- **Drawn.** Figure 7 and the graphical abstract are drawings. `scripts/fig7.py`
  draws the schematic source of Figure 7 (content, layout and labels; output
  `figures/Figure_7_schematic.png`), and the published figure,
  `figures/Figure_7_original_600dpi.png`, is an illustrated rendering of it.
  `fix_graphical_abstract_labels.py` applies the label corrections to the drawn
  graphical abstract (`figures/graphical_abstract_v52_original.png`); it is a one-off
  step, not part of `run_all.sh`, and uses a macOS system font.

## Where each table and figure comes from

Table and figure numbers below are those of the revised manuscript. Figure scripts
write into `scripts/` (or directly into `figures/`); `run_all.sh` moves them to
`figures/`, and `make_v52_figures.py` assembles the published files under
`figures/v52/`.

### Main text

| Item | Script(s) | Data |
|---|---|---|
| Table 1 | declared | — |
| Table 2 | declared | — |
| Table 3 | `rel_computation.py`, `day5_governance.py`, `rel_multiseed.py`, `governance_federated_multiseed.py`, `cross_layer_multiseed.py`, `rel_seed_robustness.py` | `tables_v52/table_03.csv` |
| Table 4 | `day5_ablation.py`, `ablation_significance.py` | `tables_v52/table_04.csv` |
| Table 5 | `governance_multiseed.py` (five seeds), `governance_federated_multiseed.py` (note, fifty draws), `day5_governance.py` (single pass, in the note) | `tables_v52/table_05.csv` |
| Table 6 | `day5_federated_crypto.py`, `federated_noise_sweep.py`, `kappa_boundary.py` | `tables_v52/table_06.csv` |
| Table 7 | `actx_routing.py` | `tables_v52/table_07.csv` |
| Table 8 | `sensitivity_sweeps.py` | `tables_v52/table_09.csv` |
| Table 9 | `proxy_architecture_sweep.py` | `tables_v52/table_10.csv` |
| Section 2.8, Section 3.2 (density-matrix reading, normalized l1 coherence) | `quantum_coherence.py` | `quantum_coherence.csv` |
| Figures 1–6 | `fig1.py`, `fig2.py`, `fig3.py`, `fig4b.py`, `fig5.py`, `fig6c.py` | schematics, no data |
| Figure 7 | drawn; schematic source `fig7.py` | — |
| Figures 8, 9 | `fig8.py`, `fig9.py` | `scenarios.csv` |
| Figure 10 | `fig10.py` | `lime_vs_shapley.csv`, `shapley_geometry.csv` |
| Figure 11 | (a) `fig11.py`, (b) `fig12.py` | `ablation_results.csv`, `ablation_significance.csv`; `operating_map.csv`, `kappa_boundary.csv` |
| Figure 12 | (a) `fig13.py`, (b) `fig14.py` | `kappa_boundary.csv`, `kappa_boundary_seedcheck.csv`; `a1_eta_sweep.csv`, `sweep_xi_four_route.csv`, `xi_operating_characteristic.csv` |
| Figure 13 | `fig16.py` | `validity_arms.csv` |
| Figure 14 | `fig_seed_convergence.py` | `seeds400.csv`, `seed_convergence.csv`, `multiseed/rel_multiseed*.csv` |

### Supplementary Materials

| Item | Script(s) | Data |
|---|---|---|
| Table S2.1 | `shapley_by_context.py`, `shapley_geometry.py` | `shapley_by_context.csv`, `shapley_geometry.csv` |
| Table S2.2 | `kappa_boundary.py` | `kappa_boundary.csv` |
| Table S2.3 | `run_misrouting_saturation.py` | `misrouting_saturation.csv` |
| Table S2.3b | `day6_ectopic_mislabel.py`, `run_misrouting_400.py` | `tables_v52/S2_3b.csv` |
| Table S2.4 | `operating_map.py` | `operating_map.csv` |
| Table S2.5 | `validity_arms.py` | `validity_arms_raw.csv` |
| Table S2.6 | `run_seed_convergence.py` | `seed_convergence.csv` |
| Table S2.7 | `run_signature_benchmark.py`, `pq_ledger.py` | `tables/table_09.csv`, `signature_benchmark.csv`, `pq_ledger.csv` |
| Table S2.8 | `ledger_cost.py` | `tables_v52/S2_8.csv` |
| Tables S3.1–S3.4 | declared (S3.4: `generator.py` writes the profiles to `pi_lookup.csv`) | — |
| Tables S4.1–S4.7b | `rel_multiseed.py`, `governance_federated_multiseed.py`, `sr_floor_sweep.py`, `xi_calibration_split.py`, `attribution_matched_reference.py` | `tables_v52/S4_*.csv` |
| Table S9.1 | `day6_ectopic_mislabel.py` | `tables_v52/table_08.csv` |
| Figure S2.1 | `fig15.py` | `misrouting_saturation.csv`, `seed_convergence.csv` |
| Figure S7.1 | `fig17.py` | `sweep_delta.csv` |
| Figure S7.2 | `fig18.py` | `sweep_eps.csv`, `sweep_sTS.csv` |

The quantities of the running text that are not in a table (for example the release
gate of Section 3.2, the counterfactual probe of Section 3.4 and the numerical floor
of Section 3.13) are written by `release_gating.py`, `gate_sensitivity.py`,
`lime_probe.py`, `sr_floor_sweep.py` and the scripts above, and are among the checks
of `verify_paper_numbers.py`.

## Numbering of the submitted version

Scripts written before the revision, `make_tables.py`, `data/tables/`, the figure
scripts `fig11.py`–`fig18.py` and several verification labels use the numbering of the
submitted version. The map to the revised manuscript:

| Object | Submitted version | Revised manuscript |
|---|---|---|
| Tables | 1 (design contract), 2 (layers), 3 (vocabulary), 4 (dominance profiles), 5 (Rel components), 6 (ablation), 7 (governance), 8 (federated/ledger), 9 (PQ signature), 10 (routing), 11 (context repair), 12 (corruption), 13 (weightings) | new Table 1 (mechanisms × operations); 1 → 2 (navigation columns → Table S3.1b); 2 → S3.2; 3 → S3.3; 4 → S3.4; 5 → 3; 6 → 4; 7 → 5; 8 → 6 (ledger-cost block → S2.8); 9 → S2.7; 10 + 11 → 7; 12 → S9.1 (misrouting block → S2.3b); 13 → 8; new Table 9 (proxy architecture) |
| Figures | 1–18 | 1–10 unchanged; 11 + 12 → Figure 11(a)(b); 13 + 14 → Figure 12(a)(b); 15 → S2.1; 16 → 13; 17 → S7.1; 18 → S7.2; new Figure 14 |
| Equations | (2.1)–(2.26) | (2.2), (2.7), (2.11)–(2.14), (2.21)–(2.23) → Supplementary (A15)–(A23); the rest renumbered (2.1)–(2.17) in order |
| Sections | 3.7.1, 3.8.1, 3.9, 3.10, 3.11, Appendices A–B | 3.7.1 → S7.1; the sweeps of 3.11 → S7.2; 3.8.1 → S9; the reversal analysis of 3.9 → S8; the permutation analysis of 3.10 → S10; Appendices → S5–S6; new 3.12 (proxy architecture); 3.12 → 3.13 (reproducibility) |

The file names under `data/tables_v52/` keep the numbering of the first revised
draft: `table_03`–`table_07` are Tables 3–7, `table_08` is Table S9.1, `table_09` is
Table 8 and `table_10` is Table 9.

## Scope

What the package establishes is internal: that the manuscript reports what its code
computed. Whether the simulator's partial traces resemble those of a deployment is the
input-realism question stated in Section 4 of the paper, and Section 2.3 gives the
predicate under which it can be measured on recorded traces without access to verified
outcomes.

## Version history and citation

Changes between versions are listed in `CHANGELOG.md`. To cite the package or the
paper, see `CITATION.cff`. Released under the MIT License (`LICENSE`).
