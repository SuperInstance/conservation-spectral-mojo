"""Benchmarks for the Conservation Spectral SDK.

Compares scalar vs SIMD Laplacian construction and measures
eigendecomposition throughput for various graph sizes.

Port note (2026-10-01): now() is gone (from std import time +
perf_counter_ns); String(x, digits)/align_right replaced with manual
rounding + padding (f-string format specs are parse errors on
1.2.0.dev2026100105).
"""

from std import time
from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian, build_laplacian_simd
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import analyze


def make_ring_graph(n: Int) -> TensionGraph:
    """Create a ring graph with n vertices."""
    var graph = TensionGraph()
    for i in range(n):
        var _ = graph.add_vertex()
    for i in range(n):
        graph.add_edge(i, (i + 1) % n, 1.0)
    return graph^


def make_random_graph(n: Int, density: Float64 = 0.3) -> TensionGraph:
    """Create a random graph with n vertices and given edge density."""
    var graph = TensionGraph()
    for i in range(n):
        var _ = graph.add_vertex()

    # Deterministic "random" for reproducibility
    var seed: UInt64 = 42
    for i in range(n):
        for j in range(n):
            if i != j:
                # Simple LCG for pseudo-random
                seed = seed * 6364136223846793005 + 1442695040888963407
                var r = Float64(seed % 1000) / 1000.0
                if r < density:
                    graph.add_edge(i, j, r + 0.1)

    return graph^


def bench_laplacian_scalar(n: Int) -> Float64:
    """Benchmark scalar Laplacian construction."""
    var graph = make_ring_graph(n)
    var start = time.perf_counter_ns()
    var lap = build_laplacian(graph, "symmetric_normalized")
    var elapsed = time.perf_counter_ns() - start
    return Float64(elapsed) / 1e6  # ms


def bench_laplacian_simd(n: Int) -> Float64:
    """Benchmark SIMD Laplacian construction."""
    var graph = make_ring_graph(n)
    var start = time.perf_counter_ns()
    var lap = build_laplacian_simd(graph, "symmetric_normalized")
    var elapsed = time.perf_counter_ns() - start
    return Float64(elapsed) / 1e6  # ms


def bench_eigendecomposition(n: Int) -> Float64:
    """Benchmark eigendecomposition."""
    var graph = make_ring_graph(n)
    var lap = build_laplacian(graph, "symmetric_normalized")
    var start = time.perf_counter_ns()
    var eigen = eigendecompose(lap)
    var elapsed = time.perf_counter_ns() - start
    return Float64(elapsed) / 1e6  # ms


def bench_full_analysis(n: Int) -> Float64:
    """Benchmark full analysis pipeline."""
    var graph = make_random_graph(n, 0.3)
    var start = time.perf_counter_ns()
    var report = analyze(graph)
    var elapsed = time.perf_counter_ns() - start
    return Float64(elapsed) / 1e6  # ms


def fmt(x: Float64, decimals: Int) -> String:
    """Manual rounding (f-string format specs are parse errors)."""
    var scale = 1.0
    for _ in range(decimals):
        scale *= 10.0
    var scaled = x * scale
    if scaled < 0.0:
        scaled -= 0.5
    else:
        scaled += 0.5
    var truncated = Int(scaled)
    var rounded = Float64(truncated) / scale
    return String(rounded)


def pad_left(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out = " " + out
    return out


def main():
    print("=== Conservation Spectral SDK — Mojo Benchmarks ===\n")

    var sizes = List[Int]()
    sizes.append(8)
    sizes.append(16)
    sizes.append(32)
    sizes.append(64)
    sizes.append(128)
    sizes.append(256)

    print("Graph Size | Scalar (ms) | SIMD (ms) | Speedup | Eigen (ms) | Full (ms)")
    print("-----------|-------------|-----------|---------|------------|----------")

    for n in sizes:
        var t_scalar = bench_laplacian_scalar(n)
        var t_simd = bench_laplacian_simd(n)
        var t_eigen = bench_eigendecomposition(n)
        var t_full = bench_full_analysis(n)
        var speedup = t_scalar / t_simd if t_simd > 0.0 else 0.0

        print(
            pad_left(String(n), 10) + " | " +
            pad_left(fmt(t_scalar, 3), 11) + " | " +
            pad_left(fmt(t_simd, 3), 9) + " | " +
            pad_left(fmt(speedup, 2), 6) + "x | " +
            pad_left(fmt(t_eigen, 3), 10) + " | " +
            pad_left(fmt(t_full, 3), 8)
        )

    print("\nBenchmark complete.")
