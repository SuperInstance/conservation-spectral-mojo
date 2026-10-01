"""Parity dump — computes the oracle's cases through the real package.

Prints machine-parseable lines for python/run_parity.py to compare.
Run: mojo run -D ASSERT=all -I . diagnostics/parity_dump.mojo
"""

from std import time
from std.math import sqrt
from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import (
    conservation_ratios, spectral_gap, cheeger_constant_approx,
    compute_spectral_fingerprint, detect_anomalies,
)


def make_graph(n: Int, directed: Bool, edges: List[Tuple[Int, Int, Float64]]) -> TensionGraph:
    var graph = TensionGraph(directed)
    for i in range(n):
        var _ = graph.add_vertex()
    for k in range(len(edges)):
        graph.add_edge(edges[k][0], edges[k][1], edges[k][2])
    return graph^


def lcg_edges(n: Int, density: Float64) -> List[Tuple[Int, Int, Float64]]:
    var edges = List[Tuple[Int, Int, Float64]]()
    var seed: UInt64 = 42
    for i in range(n):
        for j in range(n):
            if i != j:
                seed = seed * 6364136223846793005 + 1442695040888963407
                var r = Float64(seed % 1000) / 1000.0
                if r < density:
                    edges.append(Tuple(i, j, r + 0.1))
    return edges^


def dump_case(name: StringLiteral, n: Int, directed: Bool, edges: List[Tuple[Int, Int, Float64]], kind: StringLiteral):
    print("CASE", name)
    var graph = make_graph(n, directed, edges)
    print("N", n)
    print("EDGES", graph.get_edge_count())
    if n == 0:
        return
    var lap = build_laplacian(graph, kind)
    var eigen = eigendecompose(lap, 0, kind)

    # degrees
    print("DEG", end=" ")
    for i in range(n):
        print(lap.degree_vec.unsafe_load(i), end=" ")
    print()
    # Laplacian matrix (row-major)
    print("LMAT", end=" ")
    for i in range(n * n):
        print(lap.matrix.unsafe_load(i), end=" ")
    print()
    # eigenvalues
    print("EVAL", end=" ")
    for i in range(n):
        print(eigen.eigenvalues.unsafe_load(i), end=" ")
    print()
    # metrics
    print("GAP", spectral_gap(eigen))
    print("CHEEGER", cheeger_constant_approx(eigen))
    var fp = compute_spectral_fingerprint(eigen)
    print("ENTROPY", fp.spectral_entropy)
    print("EFFDIM", fp.effective_dimension)
    print("ANOM", detect_anomalies(eigen, graph, 2.0))
    # conservation ratios with vertex-index attribute (analyze's default)
    var attr = alloc[Float64](n)
    for i in range(n):
        attr.unsafe_store(i, Float64(i))
    var ratios = conservation_ratios(eigen, attr, "vertex_index")
    print("CR", end=" ")
    for k in range(len(ratios)):
        print(ratios[k].ratio, end=" ")
    print()
    attr.unsafe_free()
    # solver-independent conservation invariants
    var max_rowsum: Float64 = 0.0
    var max_asym: Float64 = 0.0
    var max_diag_err: Float64 = 0.0
    for i in range(n):
        var rs: Float64 = 0.0
        for j in range(n):
            var v = lap.matrix.unsafe_load(i * n + j)
            rs += v
            var w = lap.matrix.unsafe_load(j * n + i)
            var d = abs(v - w)
            if d > max_asym:
                max_asym = d
        if abs(rs) > max_rowsum:
            max_rowsum = abs(rs)
        if kind == "symmetric_normalized":
            var dv = abs(lap.matrix.unsafe_load(i * n + i) - 1.0)
            if dv > max_diag_err:
                max_diag_err = dv
    print("INV_ROWSUM", max_rowsum)
    print("INV_ASYM", max_asym)
    print("INV_DIAG", max_diag_err)


def main():
    var ring = List[Tuple[Int, Int, Float64]]()
    for i in range(8):
        ring.append(Tuple(i, (i + 1) % 8, 1.0))
    var path = List[Tuple[Int, Int, Float64]]()
    for i in range(5):
        path.append(Tuple(i, i + 1, 1.0))
    var star = List[Tuple[Int, Int, Float64]]()
    for i in range(1, 5):
        star.append(Tuple(0, i, 1.0))

    dump_case("ring8_unnorm", 8, False, ring, "unnormalized")
    dump_case("ring8_sym", 8, False, ring, "symmetric_normalized")
    dump_case("path6_rw", 6, False, path, "random_walk_normalized")
    dump_case("rand10_directed_sym", 10, True, lcg_edges(10, 0.3), "symmetric_normalized")
    dump_case("star5_unnorm", 5, False, star, "unnormalized")
    dump_case("two_isolated_sym", 2, False, List[Tuple[Int, Int, Float64]](), "symmetric_normalized")
    dump_case("single_vertex", 1, False, List[Tuple[Int, Int, Float64]](), "symmetric_normalized")
