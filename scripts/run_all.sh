#!/usr/bin/env bash
# Reproduces every number, table and figure of the manuscript, in order.
# Run from this directory. Paths are resolved relative to the script location,
# so the package can be unpacked anywhere.
set -euo pipefail
cd "$(dirname "$0")"

echo "== environment =="
python3 record_env.py

echo "== generation and reliability =="
python3 generator.py
python3 rel_computation.py
python3 quantum_coherence.py

echo "== ablations =="
python3 day5_ablation.py
python3 ablation_significance.py

echo "== governance =="
python3 day5_governance.py
python3 governance_multiseed.py
python3 operating_map.py

echo "== federation and ledger =="
python3 day5_federated_crypto.py
python3 federated_noise_sweep.py
python3 ledger_cost.py
python3 kappa_boundary.py
python3 pq_ledger.py
python3 run_signature_benchmark.py

echo "== context screening =="
python3 day5_ectopic.py
python3 actx_routing.py
python3 xi_four_route.py
python3 xi_operating_characteristic.py
python3 day6_ectopic_mislabel.py
python3 run_misrouting_400.py
python3 run_misrouting_saturation.py

echo "== release gate and attribution =="
python3 release_gating.py
python3 gate_sensitivity.py
python3 lime_probe.py
python3 shapley_by_context.py
python3 shapley_geometry.py

echo "== protocol sensitivity =="
python3 sensitivity_sweeps.py
python3 sigma_sweep.py
python3 run_sweep_sTS.py
python3 run_sweep_coh_thresholds.py
python3 run_repair_geometry.py
python3 validity_arms.py
python3 rel_seed_robustness.py
python3 estimator_checks.py
python3 run_seed_convergence.py

echo "== generator-level uncertainty (v1.3.0) =="
python3 generate_multiseed.py
python3 rel_multiseed.py
python3 governance_federated_multiseed.py
python3 sr_floor_sweep.py
python3 xi_calibration_split.py
python3 proxy_architecture_sweep.py
python3 attribution_matched_reference.py
python3 cross_layer_multiseed.py
python3 cross_layer_arms.py

echo "== tables and figures =="
python3 make_tables.py
for f in fig*.py; do python3 "$f"; done
mv -f Figure_*.png ../figures/ 2>/dev/null || true
python3 make_v52_figures.py
python3 make_tables_v52.py

echo "== verification =="
python3 verify_paper_numbers.py
