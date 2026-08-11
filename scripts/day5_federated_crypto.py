"""
Living Data Genome -- federated agreement + cryptographic lineage addendum.

Addresses a real gap identified during review: the paper's title claims
"Blockchain-Governed Federated Reasoning," but Section 4 as originally
scoped tested everything from a single-node perspective, with GC held at a
placeholder. This script adds two lightweight, honestly-scoped tests:

  1. Federated agreement: N=4 simulated institutional nodes independently
     apply the SAME admissibility protocol (Eq. 2.8) to a SHARED pool of
     candidate artifacts, but each node's local implementation carries a
     small independent calibration perturbation (representing realistic
     inter-institutional implementation variance), and only each node's
     binary admissibility CALL is shared -- never raw gene coordinates.
     Agreement across nodes is measured with Fleiss' kappa, a metric the
     paper already commits to (Section 1.6/2.7) but never computes.

  2. Cryptographic lineage: the governance-demo lineage events (accept,
     revoke, regenerate) are chained via SHA-256 digests (Eq. 2.21-2.22 and (A7)-(A9)
     style hash commitments), and tamper-evidence is demonstrated directly
     by mutating one record and showing the chain verification fails from
     that point forward.

Both are explicitly disclosed as simulated / lightweight instantiations
(no real multi-party network, no real PKI) -- consistent with the
simplification already declared in the decision log -- but they are no
longer placeholders: they produce real, checkable numbers.
"""

import hashlib
import json
import numpy as np
import pandas as pd

from generator import RLV_CONFIG, HEALTHCARE_CONFIG, RNG_SEED

# ---------------------------------------------------------------------------
# 1. Federated agreement via Fleiss' kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """
    rating_matrix: N items x k categories, each cell = number of raters
    assigning that item to that category (rows sum to n_raters).
    """
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)[0]  # raters per item (assumed constant)
    p_i = (np.sum(rating_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = p_i.mean()
    p_j = rating_matrix.sum(axis=0) / (N * n)
    P_e_bar = np.sum(p_j ** 2)
    if 1 - P_e_bar == 0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def federated_agreement_demo(domain_name: str, df: pd.DataFrame, config, n_nodes: int = 4,
                              node_noise_std: float = 0.015, seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(seed + 2024)
    repair = df[df.stage_t == 2]
    shared_pool = repair.sample(n=min(150, len(repair)), random_state=seed)

    ratings = np.zeros((len(shared_pool), 2), dtype=int)  # [admissible_count, inadmissible_count]
    for node in range(n_nodes):
        node_tol_delta = rng.normal(0, node_noise_std)  # each node's own small calibration drift
        calls = []
        for _, row in shared_pool.iterrows():
            context = row["context_label"]
            base_bounds = config.feasibility_bounds(context)
            g = np.array([row["g_S"], row["g_A"], row["g_D"], row["g_E"]])
            adjusted = {}
            for i, k in enumerate(["S", "A", "D", "E"]):
                lo, hi = base_bounds[k]
                mean = config.context_means[context][i]
                tol = (hi - lo) / 2 + node_tol_delta
                tol = max(tol, 0.01)
                adjusted[k] = (max(0.0, mean - tol), min(1.0, mean + tol))
            admissible = all(adjusted[k][0] <= g[i] <= adjusted[k][1] for i, k in enumerate(["S", "A", "D", "E"]))
            calls.append(admissible)
        for idx, call in enumerate(calls):
            ratings[idx, 0 if call else 1] += 1

    kappa = fleiss_kappa(ratings)
    mean_admissible_rate = ratings[:, 0].sum() / (len(shared_pool) * n_nodes)
    unanimous = np.mean((ratings[:, 0] == n_nodes) | (ratings[:, 1] == n_nodes))

    return {
        "domain": domain_name, "n_nodes": n_nodes, "n_shared_items": len(shared_pool),
        "fleiss_kappa": kappa, "mean_admissible_rate": mean_admissible_rate,
        "unanimous_rate": unanimous,
    }


# ---------------------------------------------------------------------------
# 2. Cryptographic lineage hash-chain
# ---------------------------------------------------------------------------

def digest(record: dict, prev_digest: str) -> str:
    payload = json.dumps(record, sort_keys=True) + prev_digest
    return hashlib.sha256(payload.encode()).hexdigest()


def build_chain(events: list) -> list:
    chain = []
    prev = "GENESIS"
    for ev in events:
        d = digest(ev, prev)
        chain.append({"event": ev, "digest": d, "prev": prev})
        prev = d
    return chain


def verify_chain(chain: list) -> tuple:
    prev = "GENESIS"
    for i, block in enumerate(chain):
        expected = digest(block["event"], prev)
        if expected != block["digest"] or block["prev"] != prev:
            return False, i
        prev = block["digest"]
    return True, None


def crypto_lineage_demo() -> dict:
    events = [
        {"type": "accept", "artifact_id": "A001", "t": 0},
        {"type": "certify", "artifact_id": "A001", "t": 1},
        {"type": "contest", "artifact_id": "A001", "t": 2, "reason": "regime_tightened"},
        {"type": "revoke", "artifact_id": "A001", "t": 3},
        {"type": "regenerate", "artifact_id": "A001", "successor": "A001-R", "t": 4},
        {"type": "accept", "artifact_id": "A001-R", "t": 5},
    ]
    chain = build_chain(events)
    valid_before, _ = verify_chain(chain)

    # tamper with one record (e.g., silently alter the revoke reason)
    tampered = [dict(b) for b in chain]
    tampered[3] = dict(tampered[3])
    tampered[3]["event"] = dict(tampered[3]["event"])
    tampered[3]["event"]["type"] = "accept"  # attacker tries to hide the revocation
    valid_after, break_index = verify_chain(tampered)

    return {
        "n_events": len(events), "chain_valid_before_tamper": valid_before,
        "chain_valid_after_tamper": valid_after, "tamper_detected_at_block": break_index,
    }


if __name__ == "__main__":
    df = pd.read_csv("../data/scenarios.csv")

    print("=== Federated agreement (Fleiss' kappa across 4 simulated nodes) ===")
    fed_rows = []
    for name, config in [("RLV", RLV_CONFIG), ("Healthcare", HEALTHCARE_CONFIG)]:
        sub = df[df.domain == name]
        res = federated_agreement_demo(name, sub, config, seed=RNG_SEED)
        fed_rows.append(res)
        print(f"\n{name}:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    pd.DataFrame(fed_rows).to_csv("../data/federated_agreement.csv", index=False)

    print("\n=== Cryptographic lineage hash-chain ===")
    crypto = crypto_lineage_demo()
    for k, v in crypto.items():
        print(f"  {k}: {v}")
    pd.DataFrame([crypto]).to_csv("../data/crypto_lineage_demo.csv", index=False)

    print("\nSaved: data/federated_agreement.csv, data/crypto_lineage_demo.csv")
