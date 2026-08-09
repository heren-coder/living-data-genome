"""Rebuild the permissioned ledger with real post-quantum signatures and re-measure."""
import json, hashlib, time, statistics, base64
import numpy as np, pandas as pd
from dilithium_py.ml_dsa import ML_DSA_44

H  = lambda b: hashlib.sha256(b).digest()
hx = lambda b: hashlib.sha256(b).hexdigest()
SIZES = (6, 1024, 262144)
REPEATS = 7

pk, sk = ML_DSA_44.keygen()

def payload(i):
    return {"event": ["accept","certify","contest","revoke","regenerate"][i % 5],
            "ref": hx(str(i).encode())[:64],
            "meta": {"ts": 1750000000+i, "signer": f"node{i%4}",
                     "regime": "C1", "reason": "r%03d" % (i % 7)},
            "expl": {"binding_coordinate": "Density", "slack": 0.0702,
                     "coverage_min": 0.815, "transition_stability": 0.9483,
                     "context_label": "dense_day", "regime_version": "C1",
                     "route": "proceed"}}

def record(i, scheme):
    body = payload(i)
    ser  = json.dumps(body, separators=(",", ":")).encode()
    if scheme == "keyed":
        sig = hx(str(i*7919).encode())
    else:
        sig = base64.b64encode(ML_DSA_44.sign(sk, ser)).decode()
    body["sig"] = sig
    return json.dumps(body, separators=(",", ":")).encode(), ser, sig

def levels(leaves):
    lv=[leaves]
    while len(lv[-1])>1:
        c=lv[-1]; lv.append([H(c[i]+(c[i+1] if i+1<len(c) else c[i])) for i in range(0,len(c),2)])
    return lv

def proof_path(lv, idx):
    p=[]; k=idx
    for L in lv[:-1]:
        sib = k^1 if (k^1) < len(L) else k
        p.append(L[sib]); k//=2
    return p

rows=[]
# real sign / verify cost, measured
for scheme,label in [("keyed","Keyed hash"),("mldsa","ML-DSA-44")]:
    raw,ser,sig = record(0, scheme)
    if scheme=="mldsa":
        st=[]; vt=[]
        for _ in range(REPEATS):
            t=time.perf_counter(); s2=ML_DSA_44.sign(sk,ser); st.append(time.perf_counter()-t)
            t=time.perf_counter(); ML_DSA_44.verify(pk,ser,s2); vt.append(time.perf_counter()-t)
        sign_ms=statistics.median(st)*1e3; ver_ms=statistics.median(vt)*1e3
        sigbytes=len(base64.b64decode(sig))
    else:
        st=[]; vt=[]
        for _ in range(REPEATS):
            t=time.perf_counter(); hx(ser); st.append(time.perf_counter()-t)
            t=time.perf_counter(); hx(ser); vt.append(time.perf_counter()-t)
        sign_ms=statistics.median(st)*1e3; ver_ms=statistics.median(vt)*1e3
        sigbytes=32
    for n in SIZES:
        leaves=[H(record(i%64, scheme)[0]) for i in range(min(n,64))]
        leaves=(leaves*((n//len(leaves))+1))[:n]
        t=time.perf_counter(); lv=levels(leaves); rebuild=time.perf_counter()-t
        depth=len(lv)-1
        pp=proof_path(lv,n//2)
        t=time.perf_counter()
        for _ in range(REPEATS):
            node=leaves[n//2]; k=n//2
            for sib in pp:
                node = H(node+sib) if k%2==0 else H(sib+node)
                k//=2
        ver_proof=(time.perf_counter()-t)/REPEATS
        t=time.perf_counter()
        for _ in range(REPEATS):
            node=leaves[-1]
            for sib in pp: node=H(node+sib)
        append=(time.perf_counter()-t)/REPEATS
        rows.append(dict(scheme=label,events=n,leaf_bytes=len(raw),sig_bytes=sigbytes,
                         depth=depth,proof_bytes=32*depth,
                         sign_ms=round(sign_ms,3),verify_sig_ms=round(ver_ms,3),
                         append_us=round(append*1e6,1),proof_verify_us=round(ver_proof*1e6,1),
                         rebuild_ms=round(rebuild*1e3,2)))
out=pd.DataFrame(rows); out.to_csv("pq_ledger.csv",index=False)
print(out.to_string(index=False))
