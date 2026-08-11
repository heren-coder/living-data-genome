"""Section 3.5: cost of the append-only Merkle ledger.
Timings are medians over repeated runs on a single core and are machine-dependent;
sizes and depths are exact. Outputs: data/ledger_cost.csv
"""
import hashlib, json, time
import numpy as np, pandas as pd

H = lambda b: hashlib.sha256(b).digest()
hx = lambda b: hashlib.sha256(b).hexdigest()
SIZES, REPEATS = (6, 1024, 262144), 5

def event(i):
    return {"event": ["accept","certify","contest","revoke","regenerate"][i % 5],
            "ref": hx(str(i).encode())[:64],
            "meta": {"ts": 1750000000+i, "signer": f"node{i%4}",
                     "regime": "C1", "reason": "r%03d" % (i % 7)},
            "expl": {"binding_coordinate": "Density", "slack": 0.0702,
                     "coverage_min": 0.815, "transition_stability": 0.9483,
                     "context_label": "dense_day", "regime_version": "C1",
                     "route": "proceed"},
            "sig": hx(str(i * 7919).encode())}

def levels(leaves):
    lv = [leaves]
    while len(lv[-1]) > 1:
        cur = lv[-1]
        lv.append([H(cur[i] + (cur[i+1] if i+1 < len(cur) else cur[i]))
                   for i in range(0, len(cur), 2)])
    return lv

def proof(lv, idx):
    p = []
    for d in range(len(lv)-1):
        cur = lv[d]; sib = idx ^ 1
        p.append(cur[sib] if sib < len(cur) else cur[idx]); idx //= 2
    return p

def verify(leaf, pr, idx, root):
    h = leaf
    for s in pr:
        h = H(h + s) if idx % 2 == 0 else H(s + h); idx //= 2
    return h == root

med = lambda f, n=REPEATS: float(np.median([f() for _ in range(n)]))
rows = []
for n in SIZES:
    ser = [json.dumps(event(i), sort_keys=True, separators=(",", ":")).encode()
           for i in range(n)]
    leaves = [H(s) for s in ser]
    lv = levels(leaves); root = lv[-1][0]; idx = n // 3
    pr = proof(lv, idx)
    assert verify(leaves[idx], pr, idx, root)

    def batch():
        t = time.perf_counter(); levels([H(s) for s in ser]); return (time.perf_counter()-t)*1e3
    def append_one():
        t = time.perf_counter()
        for _ in range(200):
            h = leaves[idx]
            for s in pr: h = H(h + s)
        return (time.perf_counter()-t)/200*1e6
    def vfy():
        t = time.perf_counter()
        for _ in range(2000): verify(leaves[idx], pr, idx, root)
        return (time.perf_counter()-t)/2000*1e6
    def chain():
        t = time.perf_counter(); prev = b"\x00"*32
        for s in ser: prev = H(prev + s)
        return (time.perf_counter()-t)*1e3

    rows.append(dict(events=n, leaf_bytes=int(np.mean([len(s) for s in ser])),
                     depth=len(lv)-1, proof_bytes=sum(len(x) for x in pr),
                     append_one_us=round(med(append_one), 1),
                     proof_verify_us=round(med(vfy), 1),
                     batch_rebuild_ms=round(med(batch), 2),
                     chain_verify_ms=round(med(chain), 2)))

out = pd.DataFrame(rows)
out.to_csv("../data/ledger_cost.csv", index=False)
print(out.to_string(index=False))
print("Saved: data/ledger_cost.csv")
