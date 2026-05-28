"""Benchmarks for the Conservation Spectral SDK.

Compares scalar vs SIMD Laplacian construction and measures
eigendecomposition throughput for various graph sizes.
"""

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian, build_laplacian_simd
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import analyze


fn make_ring_graph(n: Int) -> TensionGraph:
    """Create a ring graph with n vertices."""
    var graph = TensionGraph()
    for i in range(n):
        graph.add_vertex()
    for i in range(n):
        graph.add_edge(i, (i + 1) % n, 1.0)
    return graph


fn make_random_graph(n: Int, density: Float64 = 0.3) -> TensionGraph:
    """Create a random graph with n vertices and given edge density."""
    var graph = TensionGraph()
    for i in range(n):
        graph.add_vertex()

    # Deterministic "random" for reproducibility
    var seed: UInt64 = 42
    for i in range(n):
        for j in range(n):
            if i != j:
                # Simple LCG for pseudo-random
                seed = seed * 6364136223846793005 + 1442695040888963407
                let r = Float64(seed % 1000) / 1000.0
                if r < density:
                    graph.add_edge(i, j, r + 0.1)

    return graph


fn bench_laplacian_scalar(n: Int) -> Float64:
    """Benchmark scalar Laplacian construction."""
    let graph = make_ring_graph(n)
    let start = now()
    let lap = build_laplacian(graph, "symmetric_normalized")
    let elapsed = now() - start
    return Float64(elapsed) / 1e6  # ms


fn bench_laplacian_simd(n: Int) -> Float64:
    """Benchmark SIMD Laplacian construction."""
    let graph = make_ring_graph(n)
    let start = now()
    let lap = build_laplacian_simd(graph, "symmetric_normalized")
    let elapsed = now() - start
    return Float64(elapsed) / 1e6  # ms


fn bench_eigendecomposition(n: Int) -> Float64:
    """Benchmark eigendecomposition."""
    let graph = make_ring_graph(n)
    let lap = build_laplacian(graph, "symmetric_normalized")
    let start = now()
    let eigen = eigendecompose(lap)
    let elapsed = now() - start
    return Float64(elapsed) / 1e6  # ms


fn bench_full_analysis(n: Int) -> Float64:
    """Benchmark full analysis pipeline."""
    let graph = make_random_graph(n, 0.3)
    let start = now()
    let report = analyze(graph)
    let elapsed = now() - start
    return Float64(elapsed) / 1e6  # ms


fn main():
    print("=== Conservation Spectral SDK — Mojo Benchmarks ===\n")

    let sizes = [8, 16, 32, 64, 128, 256]

    print("Graph Size | Scalar (ms) | SIMD (ms) | Speedup | Eigen (ms) | Full (ms)")
    print("-----------|-------------|-----------|---------|------------|----------")

    for n in sizes:
        let t_scalar = bench_laplacian_scalar(n)
        let t_simd = bench_laplacian_simd(n)
        let t_eigen = bench_eigendecomposition(n)
        let t_full = bench_full_analysis(n)
        let speedup = t_scalar / t_simd if t_simd > 0.0 else 0.0

        print(
            String(n).align_right(10) + " | " +
            String(t_scalar, 3).align_right(11) + " | " +
            String(t_simd, 3).align_right(9) + " | " +
            String(speedup, 2).align_right(7) + "x | " +
            String(t_eigen, 3).align_right(10) + " | " +
            String(t_full, 3).align_right(8)
        )

    print("\nBenchmark complete.")
