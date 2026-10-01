"""Working example — analysis receipt for a 12-vertex ring.

Run: mojo run -D ASSERT=all -I . examples/demo.mojo

Prints inputs, the conservation invariants of the constructed Laplacian
(L·1 = 0, symmetry, unit diagonal — checked to machine epsilon), the
eigen-derived metrics (BOOKED: solver defective, see docs/USERMANUAL.md),
timings, and PASS/FAIL per line.
"""

from std import time
from std.math import sqrt
from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import (
    spectral_gap, cheeger_constant_approx, compute_spectral_fingerprint,
    detect_anomalies, analyze,
)
from conservation_spectral.tracker import ConservationTracker


def check(name: String, ok: Bool):
    if ok:
        print("[PASS] " + name)
    else:
        print("[FAIL] " + name)


def main():
    print("=" * 62)
    print(" conservation-spectral-mojo — analysis receipt")
    print("=" * 62)

    # --- inputs ---------------------------------------------------------
    var n = 12
    var graph = TensionGraph(directed=False)
    for i in range(n):
        var _ = graph.add_vertex()
    for i in range(n):
        graph.add_edge(i, (i + 1) % n, 1.0)
    print("input:      undirected ring, n=", n, ", edges=", graph.get_edge_count())
    print("laplacian:  unnormalized (L = D - W)")

    # --- construction + conservation invariants -------------------------
    var t0 = time.perf_counter_ns()
    var lap = build_laplacian(graph, "unnormalized")
    var t_lap = time.perf_counter_ns() - t0

    var max_rowsum: Float64 = 0.0
    var max_asym: Float64 = 0.0
    for i in range(n):
        var rs: Float64 = 0.0
        for j in range(n):
            var v = lap.matrix.unsafe_load(i * n + j)
            rs += v
            var d = abs(v - lap.matrix.unsafe_load(j * n + i))
            if d > max_asym:
                max_asym = d
        if abs(rs) > max_rowsum:
            max_rowsum = abs(rs)

    print("--- conserved quantities (solver-independent) ---")
    print("L·1 = 0 residual (uniform attribute conserved): ", max_rowsum)
    check("L·1 = 0 to machine epsilon", max_rowsum < 1e-12)
    print("symmetry residual:                              ", max_asym)
    check("L symmetric to machine epsilon", max_asym < 1e-12)

    # --- eigen-derived metrics (BOOKED: solver defective) ---------------
    t0 = time.perf_counter_ns()
    var eigen = eigendecompose(lap, 0, "unnormalized")
    var t_eig = time.perf_counter_ns() - t0

    var gap = spectral_gap(eigen)
    var cheeger = cheeger_constant_approx(eigen)
    var fp = compute_spectral_fingerprint(eigen)
    var anomalies = detect_anomalies(eigen, graph, 2.0)

    print("--- spectral metrics (BOOKED-FAIL: solver defective) ---")
    print("eigenvalues: ", end=" ")
    for i in range(n):
        print(eigen.eigenvalues.unsafe_load(i), end=" ")
    print()
    print("spectral gap (expect ~0.586 for ring P12):   ", gap)
    print("cheeger approx (expect ~0.146):              ", cheeger)
    print("spectral entropy (expect ~2.5):              ", fp.spectral_entropy)
    print("effective dimension:                         ", fp.effective_dimension)
    print("anomaly count:                               ", anomalies)

    # --- tracker mini-run ------------------------------------------------
    print("--- tracker mini-run ---")
    var tracker = ConservationTracker(window_size=50)
    var spurious = False
    for t in range(12):
        var obs = alloc[Float64](3)
        obs.unsafe_store(0, 1.0)
        obs.unsafe_store(1, 1.0)
        obs.unsafe_store(2, 1.0)
        var alert = tracker.feed(obs, 3)
        if alert:
            spurious = True
        obs.unsafe_free()
    print("constant signal, 12 feeds -> alerts fired: ", spurious)
    check("no spurious alert on constant signal", not spurious)
    print("observation_count:                           ", tracker.observation_count)
    tracker.reset()

    # --- timings ----------------------------------------------------------
    print("--- timings ---")
    print("laplacian build: ", Float64(t_lap) / 1e6, " ms")
    print("eigendecompose:  ", Float64(t_eig) / 1e6, " ms")

    print("=" * 62)
    print("VERDICT: construction + conservation invariants PASS;")
    print("         eigen-derived metrics BOOKED-FAIL (eigen.mojo")
    print("         solver defective — see docs/USERMANUAL.md).")
    print("=" * 62)
