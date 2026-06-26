"""Interaction-order decomposition of a combinatorially-complete DMS landscape.

Fits regularized regression models of increasing interaction order (additive ->
+pairwise -> +3rd-order) on the GB1 4-site complete binding landscape and reports
cross-validated held-out R^2 at each order. The held-out R^2 *increment* from
pairwise to 3rd-order is how much 3rd-order epistasis (background-dependence of
pairwise effects) is present and learnable from the data itself -- the quantity
a PLM-feature model would then have to predict without seeing it.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from phaseagent.mutations import parse_mutation_notation

AA = "ACDEFGHIKLMNPQRSTVWY"
aai = {a: i for i, a in enumerate(AA)}

d = pd.read_csv("/tmp/SPG1_STRSG_Wu_2016.csv").rename(columns={"mutant": "mutation_notation"})
d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
d = d[d["DMS_score"].notna() & d["mutated_sequence"].notna()].copy()
pos = sorted({int(t[1:-1]) for s in d["mutation_notation"].astype(str) for t in parse_mutation_notation(s)})
k = len(pos)

def combo(seq):
    return tuple(aai.get(seq[p - 1], -1) for p in pos)
d["combo"] = d["mutated_sequence"].astype(str).apply(combo)
d = d[d["combo"].apply(lambda c: all(x >= 0 for x in c))]
X = np.array(d["combo"].tolist())
y = d["DMS_score"].to_numpy(float)
print(f"GB1 {k}-site complete landscape: {len(y)} variants, positions {pos}")

def onehot_order(X, order):
    blocks = []
    for cp in combinations(range(k), order):
        idx = np.zeros(len(X), dtype=np.int64)
        for c in cp:
            idx = idx * 20 + X[:, c]
        blocks.append(sparse.csr_matrix((np.ones(len(X)), (np.arange(len(X)), idx)), shape=(len(X), 20 ** order)))
    return sparse.hstack(blocks).tocsr()

F1 = onehot_order(X, 1)
F2 = sparse.hstack([F1, onehot_order(X, 2)]).tocsr()
F3 = sparse.hstack([F2, onehot_order(X, 3)]).tocsr()

def cv_r2(F, y):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(y))):
        m = Ridge(alpha=10.0).fit(F[tr], y[tr])
        pred[te] = m.predict(F[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

r1, r2, r3 = cv_r2(F1, y), cv_r2(F2, y), cv_r2(F3, y)
print(f"held-out R^2:  additive={r1:.3f}   +pairwise={r2:.3f}   +3rd-order={r3:.3f}")
print(f"variance from pairwise epistasis (2nd order): {r2 - r1:.3f}")
print(f"variance from 3rd-order epistasis (background-dependence): {r3 - r2:.3f}")
print(f"unexplained even at 3rd order: {1 - r3:.3f}  (noise + 4th-order)")
