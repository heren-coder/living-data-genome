# Living Data Genome — reproducibility package

Code and generated data for the manuscript *The Living Data Genome: Ledger-Governed
Evidence Lifecycles for Accountable AI under Limited Observability* (Haluk Eren,
Fırat University, School of Civil Aviation), submitted to MDPI *Systems*.

All results in the paper come from a single deterministic pipeline over synthetic
scenarios. No external or personal data are used.

## Layout

```
scripts/    31 Python scripts: generator, reliability protocol, governance
            demonstrations and one script per figure
data/       generated CSV files, plus data/tables/ holding one CSV per
            manuscript table
figures/    the figures as they appear in the manuscript, 600 dpi PNG
```

## Requirements

Python 3.12 with the packages listed in `requirements.txt`. `dilithium-py` is
needed only by `pq_ledger.py`, which measures the post-quantum signature
substitution reported in Section 3.7.1.

```
pip install -r requirements.txt
```

## How to run

Every script is run from inside `scripts/` and reads and writes through the
relative path `../data/`.

```
cd scripts
python generator.py            # scenarios.csv, pi_lookup.csv, raw_traces.csv
python rel_computation.py      # rel_summary.csv, rel_per_event_*.csv
```

`generator.py` must be run first; everything else consumes `scenarios.csv`.
The remaining analysis scripts are independent of one another and may be run in
any order.

## What produces what

| Script | Manuscript output |
| --- | --- |
| `generator.py` | the scenario pool used everywhere; Table 4 profiles |
| `rel_computation.py` | Table 5 reliability components and aggregate |
| `day5_ablation.py` | Table 6 gene-wise ablation |
| `ablation_significance.py` | the significance tests reported under Table 6 |
| `day5_governance.py` | Table 7 governance sequences |
| `day5_federated_crypto.py` | Table 8 federated agreement and ledger cost |
| `pq_ledger.py` | Table 9 post-quantum signature substitution |
| `day5_ectopic.py` | Tables 10 and 11 context-mismatch routing and repair |
| `day6_ectopic_mislabel.py` | Table 12 deliberate context corruption |
| `shapley_by_context.py` | Section 3.4.1 attribution concentration |
| `lime_probe.py` | Section 3.4.1 local surrogate comparison |
| `kappa_boundary.py` | Section 3.7 agreement boundary |
| `operating_map.py` | Section 3.6 joint tightening and variance sweep |
| `validity_arms.py` | Section 3.10 convergent and discriminant arms |
| `fig2.py` … `fig18.py` | Figures 2 to 18, one script per figure |

Figures 1 and 7 are illustrative diagrams and are not code-generated; they are
included in `figures/` as used in the manuscript.

## Determinism

The pipeline is deterministic given a seed. `RNG_SEED` is declared in
`generator.py` and inherited by every downstream script. Where a reported
quantity is itself a property of the spread across seeds, the seed count is
raised until the estimate stops moving, as Section 3.12 records.

## License

MIT.
