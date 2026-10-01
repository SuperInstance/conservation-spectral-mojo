#!/usr/bin/env python3
"""Oracle for conservation-spectral-mojo — same formulas, numpy ground truth.

Replicates the Mojo SDK's graph/Laplacian/metric definitions EXACTLY
(out-degree for directed graphs, population variances, the same gap /
cheeger / entropy / anomaly definitions) so parity is apples-to-apples.

Eigen ground truth: numpy.linalg.eigh (symmetric path).
The Mojo solver's eigenvalues are expected to MISMATCH — that is the
booked finding; construction + invariants are expected to match to
machine epsilon.
"""

import numpy as np


def lcg_random_edges(n, density=0.3, seed0=42):
    """Replicate bench.mojo's make_random_graph edge selection (UInt64 wrap)."""
    mask = (1 << 64) - 1
    seed = seed0
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
                r = (seed % 1000) / 1000.0
                if r < density:
                    edges.append((i, j, r + 0.1))
    return edges


def build(n, directed, edges):
    W = np.zeros((n, n))
    for (s, t, w) in edges:
        W[s, t] += w
        if not directed:
            W[t, s] += w
    deg = W.sum(axis=1)  # out-degree convention (matches Mojo row sums)
    return W, deg


def laplacian(W, deg, kind):
    n = W.shape[0]
    if kind == "unnormalized":
        return np.diag(deg) - W
    if kind == "symmetric_normalized":
        dis = np.array([1.0 / np.sqrt(d) if d > 0.0 else 0.0 for d in deg])
        return np.eye(n) - (dis[:, None] * W * dis[None, :])
    if kind == "random_walk_normalized":
        dinv = np.array([1.0 / d if d > 0.0 else 0.0 for d in deg])
        return np.eye(n) - dinv[:, None] * W
    raise ValueError(kind)


def metrics(L, attr):
    """spectral_gap / cheeger / entropy / anomalies / CR — same defs as conservation.mojo."""
    n = L.shape[0]
    if n < 2:
        z = np.zeros(0)
        return dict(gap=0.0, cheeger=0.0, entropy=0.0, effdim=0.0,
                    anomalies=0, cr=np.zeros(0), evals=z)
    sym = (L + L.T) / 2.0
    evals, evecs = np.linalg.eigh(sym)  # ascending
    gap = 0.0
    for i in range(1, n - 1):
        gap = max(gap, evals[i + 1] - evals[i])
    cheeger = evals[1] / 2.0
    total = np.abs(evals).sum()
    if total < 1e-15:
        entropy = effdim = 0.0
    else:
        p = np.abs(evals) / total
        pp = p[p > 1e-15]
        entropy = float(-(pp * np.log(pp)).sum())
        effdim = float(np.exp(entropy))
    anomalies = 0
    if n >= 3:
        for k in range(min(3, n)):
            phi = evecs[:, k]
            mean = phi.sum() / n
            std = np.sqrt(((phi - mean) ** 2).sum() / n)
            if std < 1e-15:
                continue
            anomalies += int((np.abs(phi - mean) / std > 2.0).sum())
    # conservation ratios: variance of diff(phi_k * attr), population /（n-1)
    cr = []
    for k in range(n):
        if n < 2:
            cr.append(1e30)
            continue
        proj = evecs[:, k] * attr
        g = np.diff(proj)
        mean = g.sum() / (n - 1)
        cr.append(float((g * g).sum() / (n - 1) - mean * mean))
    return dict(gap=float(gap), cheeger=float(cheeger), entropy=entropy,
                effdim=effdim, anomalies=anomalies, cr=np.array(cr), evals=evals)


def invariants(L):
    """Solver-independent conservation invariants (must hold to machine eps)."""
    n = L.shape[0]
    if n == 0:
        return dict(rowsum=0.0, asym=0.0, diag_err=0.0)
    rowsum = float(np.abs(L.sum(axis=1)).max()) if n else 0.0
    asym = float(np.abs(L - L.T).max()) if n else 0.0
    return dict(rowsum=rowsum, asym=asym, diag_err=0.0)


def case_metrics(n, directed, edges, kind):
    W, deg = build(n, directed, edges)
    L = laplacian(W, deg, kind)
    attr = np.arange(n, dtype=float)
    m = metrics(L, attr)
    inv = invariants(L)
    diag_err = 0.0
    if kind == "symmetric_normalized":
        diag_err = float(np.abs(np.diag(L) - 1.0).max()) if n else 0.0
    inv["diag_err"] = diag_err
    return dict(n=n, edges=len(edges), W=W, deg=deg, L=L, **m, **inv)


def get_case(name):
    if name == "ring8_unnorm":
        e = [(i, (i + 1) % 8, 1.0) for i in range(8)]
        return case_metrics(8, False, e, "unnormalized")
    if name == "ring8_sym":
        e = [(i, (i + 1) % 8, 1.0) for i in range(8)]
        return case_metrics(8, False, e, "symmetric_normalized")
    if name == "path6_rw":
        e = [(i, i + 1, 1.0) for i in range(5)]
        return case_metrics(6, False, e, "random_walk_normalized")
    if name == "rand10_directed_sym":
        return case_metrics(10, True, lcg_random_edges(10, 0.3, 42), "symmetric_normalized")
    if name == "star5_unnorm":
        e = [(0, i, 1.0) for i in range(1, 5)]
        return case_metrics(5, False, e, "unnormalized")
    if name == "two_isolated_sym":
        return case_metrics(2, False, [], "symmetric_normalized")
    if name == "single_vertex":
        return case_metrics(1, False, [], "symmetric_normalized")
    raise KeyError(name)


CASES = ["ring8_unnorm", "ring8_sym", "path6_rw", "rand10_directed_sym",
         "star5_unnorm", "two_isolated_sym", "single_vertex"]

if __name__ == "__main__":
    for name in CASES:
        m = get_case(name)
        print(f"== {name}: n={m['n']} edges={m['edges']} gap={m['gap']:.6g} "
              f"cheeger={m['cheeger']:.6g} H={m['entropy']:.6g} anomalies={m['anomalies']}")
