"""Density-matrix reading of the cross-layer similarity matrix (Section 2.8, Section 3.2).

The three pairwise alignments of the cross-layer proxy form a symmetric similarity matrix with unit
diagonal. Divided by its trace it is positive semidefinite with unit trace, i.e. a density matrix. This
script checks those two properties, computes the l1-norm of coherence (Baumgratz, Cramer and Plenio,
2014), the sum of the absolute off-diagonal entries, and its normalization by the maximum d - 1, which
for this matrix equals the arithmetic mean of the three alignments. It reports level (geometric mean)
and balance (smallest over largest) next to it, for the two configurations and for the two illustrative
triples of Section 2.8, which carry the same normalized coherence but differ in level and balance.

Reads  data/cross_layer_coherence.csv (written by rel_computation.py)
Writes data/quantum_coherence.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data") + os.sep


def reading(label, a):
    a = np.asarray(a, float)
    S = np.array([[1, a[0], a[1]], [a[0], 1, a[2]], [a[1], a[2], 1]])
    rho = S / np.trace(S)
    eig = np.linalg.eigvalsh(rho)
    d = rho.shape[0]
    l1 = float(np.abs(rho - np.diag(np.diag(rho))).sum())
    return dict(case=label, a1=a[0], a2=a[1], a3=a[2], trace=float(np.trace(rho)), min_eigenvalue=float(eig.min()),
                l1_coherence=l1, l1_normalized=l1 / (d - 1), mean_alignment=float(a.mean()),
                level=float(np.prod(a) ** (1 / 3)), balance=float(a.min() / a.max()))


cl = pd.read_csv(D + "cross_layer_coherence.csv").set_index("domain")
rows = [reading(dom, cl.loc[dom, ["CKA_dna_rna", "CKA_dna_case", "CKA_rna_case"]].values) for dom in ["RLV", "Healthcare"]]
rows += [reading("example: even", [0.72, 0.72, 0.72]), reading("example: one pair weaker", [0.80, 0.80, 0.56])]
out = pd.DataFrame(rows)
out.to_csv(D + "quantum_coherence.csv", index=False)
print(out.round(4).to_string(index=False))
