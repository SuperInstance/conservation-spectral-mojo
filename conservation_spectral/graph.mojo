"""TensionGraph — weighted graph with vertex attributes and transition probabilities.

Mojo port (verified on-box 2026-10-01): structs are plain (no @value),
explicit __init__(out self), List (ex-DynamicVector) storage,
Pointer[Float64, MutUntrackedOrigin] (ex-UnsafePointer).
"""

from std import time


struct Edge:
    """A weighted directed edge."""
    var source: Int
    var target: Int
    var weight: Float64

    def __init__(out self, source: Int, target: Int, weight: Float64 = 1.0):
        self.source = source
        self.target = target
        self.weight = weight


struct TensionGraph:
    """Weighted directed graph with named vertex attributes.

    Stores vertices as integer IDs, edges with weights, and per-vertex
    float64 attributes in a flat array for SIMD-friendly access.
    """
    var directed: Bool
    var n_vertices: Int
    var edges: List[Edge]
    var adjacency: List[List[Int]]   # adjacency[i] = list of neighbor indices
    var adj_weights: List[List[Float64]]  # parallel weights
    var attributes: List[Float64]  # flat n_vertices × 1 for now

    def __init__(out self, directed: Bool = True):
        self.directed = directed
        self.n_vertices = 0
        self.edges = List[Edge]()
        self.adjacency = List[List[Int]]()
        self.adj_weights = List[List[Float64]]()
        self.attributes = List[Float64]()

    def add_vertex(mut self) -> Int:
        """Add a vertex. Returns its index."""
        var idx = self.n_vertices
        self.n_vertices += 1
        self.adjacency.append(List[Int]())
        self.adj_weights.append(List[Float64]())
        self.attributes.append(0.0)
        return idx

    def add_edge(mut self, source: Int, target: Int, weight: Float64 = 1.0):
        """Add a weighted edge between two vertices."""
        self.edges.append(Edge(source, target, weight))
        self.adjacency[source].append(target)
        self.adj_weights[source].append(weight)
        if not self.directed:
            self.adjacency[target].append(source)
            self.adj_weights[target].append(weight)

    def set_attribute(mut self, values: List[Float64]) raises:
        """Set vertex attributes from a flat array. Must match vertex count."""
        if len(values) != self.n_vertices:
            raise Error("Attribute length must match vertex count")
        var copy = List[Float64]()
        for i in range(len(values)):
            copy.append(values[i])
        self.attributes = copy^

    def get_edge_count(self) -> Int:
        return len(self.edges)

    def adjacency_weight(self, i: Int, j: Int) -> Float64:
        """Get weight of edge i→j (0.0 if no edge)."""
        for k in range(len(self.adjacency[i])):
            if self.adjacency[i][k] == j:
                return self.adj_weights[i][k]
        return 0.0

    def adjacency_matrix_flat(self, buf: Pointer[Float64, MutUntrackedOrigin]):
        """Write dense (n×n) adjacency matrix into buf (must be n*n Float64).

        Uses zero-copy write into caller-provided buffer.
        """
        var n = self.n_vertices
        # Zero the buffer
        for i in range(n * n):
            buf.unsafe_store(i, 0.0)

        for k in range(len(self.edges)):
            var source = self.edges[k].source
            var target = self.edges[k].target
            var weight = self.edges[k].weight
            var idx = source * n + target
            buf.unsafe_store(idx, buf.unsafe_load(idx) + weight)
            if not self.directed:
                var idx2 = target * n + source
                buf.unsafe_store(idx2, buf.unsafe_load(idx2) + weight)

    def degree(self, vertex: Int) -> Float64:
        """Sum of outgoing edge weights for a vertex."""
        var total: Float64 = 0.0
        var weights = self.adj_weights[vertex]
        for k in range(len(weights)):
            total += weights[k]
        return total

    @staticmethod
    def build_from_transitions(
        transitions: List[Tuple[Int, Int]],
        directed: Bool = True,
    ) -> TensionGraph:
        """Build a graph from a sequence of (from, to) transitions.

        Each transition increments the edge weight by 1.0.
        Aggregates duplicate edges.
        """
        var graph = TensionGraph(directed)

        # Find max vertex index to pre-allocate
        var max_v: Int = 0
        for t in transitions:
            if t[0] > max_v:
                max_v = t[0]
            if t[1] > max_v:
                max_v = t[1]

        # Add vertices 0..max_v
        for _ in range(max_v + 1):
            graph.add_vertex()

        # Count transitions into a simple map (src*max_n + tgt) → weight
        var n = max_v + 1
        var counts = List[Float64]()
        for _ in range(n * n):
            counts.append(0.0)

        for t in transitions:
            var src = t[0]
            var tgt = t[1]
            var idx = src * n + tgt
            counts[idx] = counts[idx] + 1.0

        # Add edges with aggregated weights
        for src in range(n):
            for tgt in range(n):
                var w = counts[src * n + tgt]
                if w > 0.0:
                    graph.add_edge(src, tgt, w)

        return graph^
