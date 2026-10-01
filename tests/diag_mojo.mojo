from std import time
from std.math import sqrt
from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose

def dump_evals(n: Int, directed: Bool):
    var graph = TensionGraph(directed)
    for i in range(n):
        graph.add_vertex()
    for i in range(n - 1):
        graph.add_edge(i, i + 1, 1.0)
    var lap = build_laplacian(graph, "unnormalized")
    print("degrees:", end=" ")
    for i in range(n):
        print(lap.degree_vec.unsafe_load(i), end=" ")
    print()
    var eigen = eigendecompose(lap)
    print("evals:", end=" ")
    for i in range(n):
        print(eigen.eigenvalues.unsafe_load(i), end=" ")
    print()

def main():
    print("-- directed path P4 (test as written):")
    dump_evals(4, True)
    print("-- undirected path P4 (test intent):")
    dump_evals(4, False)
    print("-- directed path P3 degrees (test as written):")
    var g = TensionGraph()
    for i in range(3):
        g.add_vertex()
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    var lap = build_laplacian(g, "unnormalized")
    for i in range(3):
        print("d", i, "=", lap.degree_vec.unsafe_load(i), " L[0][1]=", lap.load_element(lap.matrix, 0, 1))
