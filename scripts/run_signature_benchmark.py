"""Three-way signature comparison behind Table 9.

Section 3.7.1 prices the post-quantum substitution against a classical baseline
rather than against the keyed hashes of the demonstration, because the comparison
a deployment faces is the migration from classical signatures. All three arms sign
the same canonical event record. Writes signature_benchmark.csv.
"""
import json, hashlib, base64, time, statistics, pandas as pd
from dilithium_py.ml_dsa import ML_DSA_44
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPEATS = 25
hx = lambda b: hashlib.sha256(b).hexdigest()

BODY = {"event": "accept", "ref": hx(b"0")[:64],
        "meta": {"ts": 1750000000, "signer": "node0", "regime": "C1", "reason": "r000"},
        "expl": {"binding_coordinate": "Density", "slack": 0.0702, "coverage_min": 0.815,
                 "transition_stability": 0.9483, "context_label": "dense_day",
                 "regime_version": "C1", "route": "proceed"}}
SER = json.dumps(BODY, separators=(",", ":")).encode()

def record_bytes(sig):
    s = base64.b64encode(sig).decode() if isinstance(sig, bytes) else sig
    return len(json.dumps({**BODY, "sig": s}, separators=(",", ":")).encode())

def bench(name, sign, verify, sig_bytes):
    st, vt = [], []
    for _ in range(REPEATS):
        t = time.perf_counter(); s = sign(); st.append(time.perf_counter() - t)
        t = time.perf_counter(); verify(s);   vt.append(time.perf_counter() - t)
    return dict(scheme=name, sig_bytes=sig_bytes, record_bytes=record_bytes(s),
                sign_ms=round(statistics.median(st) * 1e3, 3),
                verify_ms=round(statistics.median(vt) * 1e3, 3), repeats=REPEATS)

rows = [bench("Keyed hash", lambda: hx(SER), lambda s: hx(SER), 32)]

k = Ed25519PrivateKey.generate(); p = k.public_key()
rows.append(bench("Ed25519", lambda: k.sign(SER), lambda s: p.verify(s, SER), 64))

pk, sk = ML_DSA_44.keygen()
rows.append(bench("ML-DSA-44", lambda: ML_DSA_44.sign(sk, SER),
                  lambda s: ML_DSA_44.verify(pk, SER, s), 2420))

t = pd.DataFrame(rows)
t.to_csv("../data/signature_benchmark.csv", index=False)
print(t.to_string(index=False))
