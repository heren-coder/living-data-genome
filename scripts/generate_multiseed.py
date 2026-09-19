"""
generate_multiseed.py -- generator-level resampling.

Why this script exists
----------------------
Every downstream script reads a single scenarios.csv produced with RNG_SEED=42.
The "five seeds" of the submitted manuscript therefore vary only the
case-facing noise draw and the baseline split (see rel_seed_robustness.py);
gene coordinates, TS and SR are identical across those seeds. Uncertainty in
quantities affected by sampling requires resampling the DATA, which is what
this script does.

It calls generate_domain_dataset() unchanged, once per generator seed, and
writes seed-indexed files. Nothing in the simulation mechanism is modified.

Seed convention (record this in Supplementary S3):
    generator seed k = 0..N_SEEDS-1
    RLV        : seed = RNG_SEED + 1000*k
    Healthcare : seed = RNG_SEED + 1000*k + 1
so k = 0 reproduces the submitted scenarios.csv byte-for-byte (verified below),
and the stride of 1000 keeps generator seeds disjoint from the evaluation seeds
RNG_SEED + 100*i used by the existing five-seed scripts.

Outputs (data/multiseed/):
    scenarios_seed{k:02d}.csv , raw_traces_seed{k:02d}.csv   for each k
    multiseed_manifest.csv   -- one row per seed: seeds used, row counts,
                                stage-wise admissibility, SR=0 event counts
Runtime: pure NumPy, a few seconds for 50 seeds.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generator import (RNG_SEED, RLV_CONFIG, HEALTHCARE_CONFIG,  # noqa: E402
                       generate_domain_dataset)

N_SEEDS = int(os.environ.get("LDG_N_SEEDS", 50))
STRIDE = 1000
D = os.path.join(_HERE, "..", "data") + os.sep
OUT = os.path.join(D, "multiseed") + os.sep
os.makedirs(OUT, exist_ok=True)


def gen_seeds(k: int):
    return RNG_SEED + STRIDE * k, RNG_SEED + STRIDE * k + 1


def generate(k: int):
    s_rlv, s_hc = gen_seeds(k)
    sink = []
    rlv = generate_domain_dataset(RLV_CONFIG, n_events=200, seed=s_rlv, trace_sink=sink)
    hc = generate_domain_dataset(HEALTHCARE_CONFIG, n_events=200, seed=s_hc, trace_sink=sink)
    return pd.concat([rlv, hc], ignore_index=True), pd.DataFrame(sink), (s_rlv, s_hc)


def summarize(k, seeds, df):
    row = {"k": k, "seed_rlv": seeds[0], "seed_hc": seeds[1], "n_rows": len(df)}
    for dom in ["RLV", "Healthcare"]:
        d = df[df.domain == dom]
        for t in (0, 1, 2):
            row[f"{dom}_Ag_stage{t}"] = round(float(d[d.stage_t == t]["A_g"].mean()), 4)
        sr = d[d.stage_t == 2].groupby("event_id")["A_g"].mean()
        row[f"{dom}_SR_mean"] = round(float(sr.mean()), 4)
        row[f"{dom}_SR_zero_events"] = int((sr == 0).sum())
    return row


if __name__ == "__main__":
    t0 = time.time()
    manifest = []
    for k in range(N_SEEDS):
        df, traces, seeds = generate(k)
        df.to_csv(OUT + f"scenarios_seed{k:02d}.csv", index=False)
        traces.to_csv(OUT + f"raw_traces_seed{k:02d}.csv", index=False)
        manifest.append(summarize(k, seeds, df))
    man = pd.DataFrame(manifest)
    man.to_csv(OUT + "multiseed_manifest.csv", index=False)

    # Backward-compatibility check: k = 0 must equal the submitted scenarios.csv
    ref = pd.read_csv(D + "scenarios.csv")
    new = pd.read_csv(OUT + "scenarios_seed00.csv")
    try:
        pd.testing.assert_frame_equal(ref, new, check_exact=False, rtol=0, atol=1e-12)
        compat = "PASS: scenarios_seed00.csv == submitted scenarios.csv"
    except AssertionError as e:
        compat = "FAIL: seed 0 does not reproduce scenarios.csv -- " + str(e).splitlines()[0]

    print(f"=== generate_multiseed: {N_SEEDS} seeds in {time.time() - t0:.1f}s ===")
    print(compat)
    cols = ["k", "RLV_SR_mean", "RLV_SR_zero_events", "Healthcare_SR_mean", "Healthcare_SR_zero_events"]
    print(man[cols].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    print(f"Saved: {OUT}scenarios_seedNN.csv, raw_traces_seedNN.csv, multiseed_manifest.csv")
