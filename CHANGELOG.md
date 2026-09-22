## v1.3.9 — Supplementary sections follow the order of the main text, September 2026

No script, data file, table, figure or number changed. README only:
- Supplementary S8 and S9 exchange places in the manuscript so that they follow Sections 3.8 and 3.9: the
  context-corruption experiment is now Supplementary S8 with Table S8.1, and the reversal analysis of
  Section 3.9 is S9. The table map, the section map and the note on data/tables_v52/table_08.csv are updated.
- README version and CITATION.cff version 1.3.9.

## v1.3.8 — minor revision of the manuscript (estimator notes), September 2026

- New scripts/estimator_checks.py writes data/estimator_checks.csv: the triadic alignment of the single pass
  (Table 3) against the first of the five evaluation seeds, which reproduces it, and their five-seed mean and
  standard deviation (Table 4 headings; 0.002 and 0.008); and the stability term of Equation (2.11) as the
  arithmetic mean of the two governed transitions (0.917) against their geometric mean (0.915). run_all.sh
  runs it after rel_seed_robustness.py.
- verify_paper_numbers.py: ten checks for the above. 282 checks in total.
- README: version, check count and the row for the new script.

No other script, data file, table or figure changed.


## v1.3.7 — minor revision of the manuscript, September 2026

- New scripts/quantum_coherence.py writes data/quantum_coherence.csv: the density-matrix reading of the
  cross-layer similarity matrix (unit trace, positive semidefinite), its l1-norm of coherence and the
  normalization that equals the arithmetic mean of the three alignments, next to level and balance, for
  both configurations (0.717 / 0.755) and for the two illustrative triples of Section 2.8. run_all.sh runs
  it after rel_computation.py.
- verify_paper_numbers.py: eighteen checks for the above. 272 checks in total.
- data/tables_v52/table_03.csv: the governance-coherence row carries the manuscript's label, naming the
  discard arm and its single-pass revoke rate. No value changed.
- README: version, check count and the row for the new script.

No other script, data file, table or figure changed.


## v1.3.6 — version of the revised manuscript, September 2026

Versions 1.3.0 to 1.3.5 below were internal steps of the revision; they are released
together as v1.3.6, the version reported in the revised manuscript.

Numbers and tables
- Table 3 single-file aggregate rows on one route: the idealized single-file row is written
  on the component route (0.869 / 0.877), the same route as the measured row (0.800 / 0.812)
  and the 50-draw rows; the per-event mean (0.857 / 0.861), to which the per-event bootstrap
  interval belongs, keeps its own labelled row.
- data/tables_v52/table_03.csv follows Table 3 of the manuscript row by row (transition
  stability split by stage, governance coherence, bootstrap interval, supervised baseline).
- data/tables_v52/table_05.csv: the discard coherence is reported from the same five-seed run
  as the other rows of Table 5 (0.706 ± 0.032 / 0.678 ± 0.019), and its columns are labelled
  five seeds; the single-pass value 0.717 / 0.736 remains the one entering Table 3.
- Supplementary S4 table CSVs are written at full precision, so every printed value rounds
  from them directly.
- sensitivity_sweeps.py reads the discard coherence from governance_demo_results.csv instead
  of a typed constant; sweep_weights.csv changes below 1e-6 and no printed value changes.
- New scripts/shapley_geometry.py writes data/shapley_geometry.csv (region geometry behind
  Corollary C2), previously supplied as a data file; its output is byte-identical to that file.

Figures
- New scripts/fig_seed_convergence.py writes figures/v52/Figure_14.png (evaluation-seed
  convergence of the aggregate against the spread across fifty generator draws).
- Figure 1 redrawn in matplotlib (new scripts/fig1.py); Figures 3, 4, 5, 9, 10, 11, 12 and
  Supplementary Figure S2.1 redrawn for legibility only, with no data or number changed.
- scripts/fig7.py is the schematic source of Figure 7, with the published labels (Protocol
  reliability); it writes Figure_7_schematic.png. The published Figure 7 is the drawn
  rendering figures/Figure_7_original_600dpi.png.
- figures/Figure_1_from_pages.png, which nothing referenced, removed.

Verification
- verify_paper_numbers.py: six checks for the Table 3 routes, sixteen for the seed spread
  (Sections 3.3 and 3.13, Figure 14), every printed cell of Table 9 and the discard coherence
  of Table 5. 254 checks in total. (The first v1.3.6 draft stated 185; the count was 184,
  the final summary line having been counted as a check.)

Environment and housekeeping
- requirements.txt pins the environment that produced the published numbers; new
  scripts/record_env.py writes all eight package versions to data/env.json at the start of
  run_all.sh.
- Comments and console output in English throughout; docstrings describe the code as it
  stands; unused variables, duplicate imports and one unused branch removed; .DS_Store ignored.
- README rewritten: every table and figure of the revised manuscript and Supplement mapped to
  its scripts and data files. CITATION.cff names the paper as the preferred citation.


## v1.3.5 — documentation aligned with the resubmitted manuscript, September 2026

No script, figure or number changed. README only:
- the table, figure and section map now reflects the final numbering of the
  revised manuscript: the context-corruption experiment (Section 3.8.1 of the
  submitted version) is Supplementary S9 with Table S9.1, the weighting sweep is
  Table 8 and the proxy-architecture sensitivity Table 9; the reversal analysis
  of Section 3.9 is S8 and the permutation qualification of Section 3.10 is S10.
- the check count of verify_paper_numbers.py is stated as 162 (it had read 141,
  the count before v1.3.2).
- the file names under data/tables_v52/ are explained against the final numbering.
- CITATION.cff version 1.3.5.

## v1.3.4 — graphical abstract coordinate names, September 2026

- figures/graphical_abstract_v52.png — the healthcare coordinates are now spelled
  as Table 4 spells them: AccessEscalationRate, ConcurrentRecordDensity and
  AccessControlRegime, against the abbreviated AccessEscalation, RecordDensity
  and AccessControl the poster carried. The full names do not fit at the original
  size, so the row is redrawn one step smaller in PT Sans; no other pixel changes.
- scripts/fix_graphical_abstract_labels.py — performs that edit, so the one change
  made to a hand-drawn figure is reproducible from figures/graphical_abstract_v52_original.png.


## v1.3.2 — cross-layer proxy quantified, September 2026

The cross-layer coherence proxy of Section 2.8 was the one quantity of the
manuscript that the revision's fifty-draw treatment had not reached, and it had
never been put through the validity arms of Section 3.10. Both gaps are closed
here. The proxy itself is unchanged; compute_cross_layer() is called as it stands.

- scripts/cross_layer_multiseed.py — the proxy over 50 generator draws. Draw
  k = 0 reproduces the submitted single-file values exactly (0.715 / 0.834 and
  0.754 / 0.869). Over the fifty draws the level component averages 0.704 and
  0.720 and the balance component 0.826 and 0.838, and at the declared
  thresholds the coherence flag is raised on 52 percent of draws in the
  red-light violation configuration and 76 percent in the healthcare vignette.
  The submitted file therefore sits at the favourable end of the distribution
  and the declared level threshold sits at the RLV mean, which quantifies the
  narrow margin the submitted text reported qualitatively. The flag enters no
  reported quantity, so no other number changes.
- scripts/cross_layer_arms.py — the proxy across the validity arms. It is exactly
  invariant under the two discriminant transformations, falls by 0.309 and 0.338
  in level under semantic misalignment against the aggregate's 0.124 and 0.130,
  falls by 0.119 and 0.101 under loss of releasability, and does not move at all
  under transition instability. Balance responds about 1.5 times as strongly as
  level in every arm, which is the empirical counterpart of Proposition P4.
- scripts/verify_paper_numbers.py — 16 checks added for the above. 162 in total.
- scripts/run_all.sh — runs the two new scripts.
- figures/Figure_7_original_600dpi.png and figures/graphical_abstract_v52.png — labels
  'Reliability' -> 'Protocol reliability' (panel 4 header and bottom band); the graphical
  abstract is hand-drawn and has no script.


## v1.3.1 — consistency corrections, September 2026

Three reporting inconsistencies found while auditing the v52 revision. None was
requested by a reviewer and none reverses a conclusion; two of them raise the
governance contribution the manuscript claims.

- make_tables_v52.py — Table 3 reported the idealized aggregate from the mean of
  the per-event aggregates and the measured aggregate from the weighted geometric
  mean of the component means, so their difference mixed two estimators. Both rows
  are now computed on the component route: idealized 0.861 / 0.857 against measured
  0.789 / 0.785, and the governance contribution is 0.072 in both configurations
  against the 0.057 / 0.052 the mixed pair implied.
- make_tables_v52.py — Table 5 carried the discard coherence from the single-file
  run next to a 50-draw revoke rate, which broke the exact identity
  GC = 1 - revoke across rows of one table. The 50-draw block now carries the
  coherence and the baseline aggregate: 0.705 / 0.704 against revoke 0.295 / 0.296.
- fig12.py — the operating-map band was drawn at the median across tightening
  columns of the first grid point exceeding CV = 0.5 on this figure's 5-seed coarse
  sweep (17.5 / 20 percent), while the text and Table 6 report the interpolated
  CV = 0.5 crossing on the 25-seed sweep (16.3 / 17.0 percent). The band now reads
  the declared boundary.
- verify_paper_numbers.py — checks added for the measured aggregate and for the
  governance contribution, neither of which was previously verified. 146 checks in total.

## v1.3.0 — revision for Systems (systems-4529608), September 2026

Added (no existing script or number changed):
- scripts/generate_multiseed.py — 50 generator-level resamples (seed 42+1000k); k=0 reproduces scenarios.csv byte-for-byte
- scripts/rel_multiseed.py — Rel components, baseline and gene ablations over 50 draws with bootstrap CIs (Table 3 of v52, S4.1–S4.2)
- scripts/governance_federated_multiseed.py — Table 5/6 (v52) governance and Fleiss κ over 50 draws, pooled Clopper–Pearson (S4.3–S4.4)
- scripts/sr_floor_sweep.py — quantitative effect of the SR numerical floor, ε and δ sweeps (Section 3.12, S4.5)
- scripts/xi_calibration_split.py — context-mismatch tolerance declared on calibration draws, evaluated held-out (Section 3.8, S4.6)
- scripts/proxy_architecture_sweep.py — sensitivity of Rel to the trace-to-gene proxy architecture (Section 3.13, Table 10 of v52)
- scripts/attribution_matched_reference.py — 2×2 matched-reference comparison of attribution methods (Section 3.4.1, S4.7)
- data/multiseed/ — summary CSVs written by the above (per-draw scenario files are regenerated by generate_multiseed.py, 7 s)

- scripts/make_v52_figures.py — assembles the revised figure files (two-panel Figures 11–12, Supplementary S2.1/S7.1/S7.2) under figures/v52/
- scripts/fig3.py — Figure 3 redrawn for legibility (all text ≥6 pt, taller canvas)
- scripts/verify_paper_numbers.py — checks added for the revision's headline numbers
- scripts/make_tables_v52.py — writes the revised tables (3–10, S2.3b, S2.8, S4.1–S4.7b) under data/tables_v52/
- scripts/run_all.sh — runs the multiseed group, make_v52_figures.py and make_tables_v52.py
- .gitignore — per-draw scenario/trace files (regenerated in seconds) and __pycache__
- README — revision section with the v51→v52 table/figure/equation map
