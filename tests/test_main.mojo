"""Tests for the Conservation Spectral SDK."""

from conservation_spectral.graph import TensionGraph, Edge
from conservation_spectral.laplacian import Laplacian, build_laplacian, build_laplacian_simd
from conservation_spectral.eigen import EigenDecomposition, eigendecompose
from conservation_spectral.conservation import (
    conservation_ratio, conservation_ratios, spectral_gap,
    cheeger_constant_approx, analyze, ConservationReport,
)
from conservation_spectral.tracker import ConservationTracker, Alert


fn test_graph_basic() -> Bool:
    """Test basic graph construction."""
    var graph = TensionGraph()
    let v0 = graph.add_vertex()
    let v1 = graph.add_vertex()
    let v2 = graph.add_vertex()

    if graph.n_vertices != 3:
        return False

    graph.add_edge(v0, v1, 2.0)
    graph.add_edge(v1, v2, 3.0)
    graph.add_edge(v0, v2, 1.0)

    if graph.get_edge_count() != 3:
        return False

    # Check adjacency weight
    if graph.adjacency_weight(0, 1) != 2.0:
        return False
    if graph.adjacency_weight(1, 0) != 0.0:  # directed
        return False

    return True


fn test_graph_undirected() -> Bool:
    """Test undirected graph."""
    var graph = TensionGraph(directed=False)
    let v0 = graph.add_vertex()
    let v1 = graph.add_vertex()
    let v2 = graph.add_vertex()

    graph.add_edge(v0, v1, 2.0)
    graph.add_edge(v1, v2, 3.0)

    if graph.get_edge_count() != 2:
        return False

    # In undirected graph, adjacency should be symmetric
    if graph.adjacency_weight(0, 1) != 2.0:
        return False
    if graph.adjacency_weight(1, 0) != 2.0:
        return False

    return True


fn test_laplacian_unnormalized() -> Bool:
    """Test unnormalized Laplacian construction."""
    var graph = TensionGraph()
    let v0 = graph.add_vertex()
    let v1 = graph.add_vertex()
    let v2 = graph.add_vertex()

    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 2, 1.0)

    let lap = build_laplacian(graph, "unnormalized")

    if lap.n != 3:
        return False

    # Diagonal should be degrees: d0=1, d1=2, d2=1
    if abs(lap.load_element(lap.matrix, 0, 0) - 1.0) > 1e-10:
        return False
    if abs(lap.load_element(lap.matrix, 1, 1) - 2.0) > 1e-10:
        return False

    # Off-diagonal should be -w
    if abs(lap.load_element(lap.matrix, 0, 1) - (-1.0)) > 1e-10:
        return False

    return True


fn test_laplacian_normalized() -> Bool:
    """Test symmetric normalized Laplacian."""
    var graph = TensionGraph()
    let v0 = graph.add_vertex()
    let v1 = graph.add_vertex()
    let v2 = graph.add_vertex()

    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 2, 1.0)

    let lap = build_laplacian(graph, "symmetric_normalized")

    # Diagonal should be 1.0 for normalized
    if abs(lap.load_element(lap.matrix, 0, 0) - 1.0) > 1e-10:
        return False
    if abs(lap.load_element(lap.matrix, 2, 2) - 1.0) > 1e-10:
        return False

    return True


fn test_laplacian_simd() -> Bool:
    """Test SIMD Laplacian matches scalar version."""
    var graph = TensionGraph()
    for i in range(8):
        graph.add_vertex()

    # Create a ring graph
    for i in range(8):
        graph.add_edge(i, (i + 1) % 8, 1.0)

    let lap_scalar = build_laplacian(graph, "unnormalized")
    let lap_simd = build_laplacian_simd(graph, "unnormalized")

    # Check they produce the same result
    for i in range(8):
        for j in range(8):
            let s = lap_scalar.load_element(lap_scalar.matrix, i, j)
            let v = lap_simd.load_element(lap_simd.matrix, i, j)
            if abs(s - v) > 1e-10:
                return False

    return True


fn test_eigendecomposition() -> Bool:
    """Test eigendecomposition of a simple graph."""
    var graph = TensionGraph()
    for i in range(4):
        graph.add_vertex()

    # Path graph: 0-1-2-3
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 1.0)

    let lap = build_laplacian(graph, "unnormalized")
    let eigen = eigendecompose(lap)

    # Should have 4 eigenvalues
    if eigen.n != 4:
        return False

    # First eigenvalue should be ~0 (connected graph)
    if abs(eigen.eigenvalues.load(0)) > 1e-6:
        return False

    # Eigenvalues should be sorted ascending
    for i in range(3):
        if eigen.eigenvalues.load(i) > eigen.eigenvalues.load(i + 1) + 1e-10:
            return False

    return True


fn test_conservation_ratio() -> Bool:
    """Test conservation ratio computation."""
    var graph = TensionGraph()
    for i in range(5):
        graph.add_vertex()

    # Ring graph
    for i in range(5):
        graph.add_edge(i, (i + 1) % 5, 1.0)

    let lap = build_laplacian(graph)
    let eigen = eigendecompose(lap)

    var attr = UnsafePointer[Float64].alloc(5)
    for i in range(5):
        attr.store(i, Float64(i))

    let cr = conservation_ratio(eigen, attr, 0)
    # Ratio should be a finite non-negative number
    if cr < 0.0 or cr > 1e10:
        attr.free()
        return False

    attr.free()
    return True


fn test_spectral_gap() -> Bool:
    """Test spectral gap computation."""
    var graph = TensionGraph()
    for i in range(6):
        graph.add_vertex()

    for i in range(6):
        graph.add_edge(i, (i + 1) % 6, 1.0)

    let lap = build_laplacian(graph)
    let eigen = eigendecompose(lap)
    let gap = spectral_gap(eigen)

    # Spectral gap should be positive
    if gap < 0.0:
        return False

    return True


fn test_full_analysis() -> Bool:
    """Test the full analyze() pipeline."""
    var graph = TensionGraph()
    for i in range(6):
        graph.add_vertex()

    for i in range(6):
        graph.add_edge(i, (i + 1) % 6, 1.0)

    let report = analyze(graph)

    # Should have ratios for all eigenvectors
    if report.ratios.size != 6:
        return False

    # Spectral gap should be non-negative
    if report.spectral_gap < 0.0:
        return False

    # Cheeger constant should be non-negative
    if report.cheeger_constant < 0.0:
        return False

    # Fingerprint should have valid entropy
    if report.fingerprint.spectral_entropy < 0.0:
        return False

    return True


fn test_tracker() -> Bool:
    """Test the conservation tracker."""
    var tracker = ConservationTracker(window_size=50)

    # Feed some observations
    for t in range(20):
        var obs = UnsafePointer[Float64].alloc(4)
        obs.store(0, sin(Float64(t) * 0.1))
        obs.store(1, cos(Float64(t) * 0.1))
        obs.store(2, sin(Float64(t) * 0.2))
        obs.store(3, 1.0)
        let _ = tracker.feed(obs, 4)
        obs.free()

    # Should have processed observations
    if tracker.observation_count != 20:
        return False

    return True


fn test_build_from_transitions() -> Bool:
    """Test building graph from transitions."""
    var transitions = DynamicVector[Tuple[Int, Int]]()
    transitions.push_back(Tuple(0, 1))
    transitions.push_back(Tuple(1, 2))
    transitions.push_back(Tuple(2, 0))
    transitions.push_back(Tuple(0, 1))  # duplicate

    let graph = TensionGraph.build_from_transitions(transitions)

    if graph.n_vertices != 3:
        return False

    # Should have 3 edges (0→1 aggregated weight=2.0)
    if graph.get_edge_count() != 3:
        return False

    # Check weight of 0→1 is 2.0
    if abs(graph.adjacency_weight(0, 1) - 2.0) > 1e-10:
        return False

    return True


fn main():
    print("=== Conservation Spectral SDK — Mojo Tests ===\n")

    var passed = 0
    var failed = 0

    fn run_test(name: StringRef, test_fn: fn() -> Bool):
        nonterminal
        let result = test_fn()
        if result:
            print("[PASS] " + name)
            passed += 1
        else:
            print("[FAIL] " + name)
            failed += 1

    run_test("Graph basic construction", test_graph_basic)
    run_test("Undirected graph", test_graph_undirected)
    run_test("Unnormalized Laplacian", test_laplacian_unnormalized)
    run_test("Normalized Laplacian", test_laplacian_normalized)
    run_test("SIMD Laplacian", test_laplacian_simd)
    run_test("Eigendecomposition", test_eigendecomposition)
    run_test("Conservation ratio", test_conservation_ratio)
    run_test("Spectral gap", test_spectral_gap)
    run_test("Full analysis pipeline", test_full_analysis)
    run_test("Conservation tracker", test_tracker)
    run_test("Build from transitions", test_build_from_transitions)

    print("\n=== Results: " + String(passed) + " passed, " + String(failed) + " failed ===")
    if failed > 0:
        print("SOME TESTS FAILED")
    else:
        print("ALL TESTS PASSED")
