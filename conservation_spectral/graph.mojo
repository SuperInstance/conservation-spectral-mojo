"""TensionGraph — weighted graph with vertex attributes and transition probabilities.

Uses Mojo's owned/inout semantics for zero-copy edge manipulation and
DynamicVector for growable edge/vertex storage.
"""

from collections.dynamic_vector import DynamicVector


@value
struct Edge:
    """A weighted directed edge."""
    var source: Int
    var target: Int
    var weight: Float64

    fn __init__(inout self, source: Int, target: Int, weight: Float64 = 1.0):
        self.source = source
        self.target = target
        self.weight = weight


@value
struct TensionGraph:
    """Weighted directed graph with named vertex attributes.

    Stores vertices as integer IDs, edges with weights, and per-vertex
    float64 attributes in a flat array for SIMD-friendly access.
    """
    var directed: Bool
    var n_vertices: Int
    var edges: DynamicVector[Edge]
    var adjacency: DynamicVector[DynamicVector[Int]]   # adjacency[i] = list of neighbor indices
    var adj_weights: DynamicVector[DynamicVector[Float64]]  # parallel weights
    var attributes: DynamicVector[Float64]  # flat n_vertices × 1 for now

    fn __init__(inout self, directed: Bool = True):
        self.directed = directed
        self.n_vertices = 0
        self.edges = DynamicVector[Edge]()
        self.adjacency = DynamicVector[DynamicVector[Int]]()
        self.adj_weights = DynamicVector[DynamicVector[Float64]]()
        self.attributes = DynamicVector[Float64]()

    fn add_vertex(inout self) -> Int:
        """Add a vertex. Returns its index."""
        let idx = self.n_vertices
        self.n_vertices += 1
        self.adjacency.push_back(DynamicVector[Int]())
        self.adj_weights.push_back(DynamicVector[Float64]())
        self.attributes.push_back(0.0)
        return idx

    fn add_edge(inout self, source: Int, target: Int, weight: Float64 = 1.0):
        """Add a weighted edge between two vertices."""
        self.edges.push_back(Edge(source, target, weight))
        self.adjacency[source].push_back(target)
        self.adj_weights[source].push_back(weight)
        if not self.directed:
            self.adjacency[target].push_back(source)
            self.adj_weights[target].push_back(weight)

    fn set_attribute(inout self, values: DynamicVector[Float64]):
        """Set vertex attributes from a flat array. Must match vertex count."""
        if values.size != self.n_vertices:
            raise("Attribute length must match vertex count")
        self.attributes = values

    fn get_edge_count(self) -> Int:
        return self.edges.size

    fn adjacency_weight(self, i: Int, j: Int) -> Float64:
        """Get weight of edge i→j (0.0 if no edge)."""
        let neighbors = self.adjacency[i]
        let weights = self.adj_weights[i]
        for k in range(neighbors.size):
            if neighbors[k] == j:
                return weights[k]
        return 0.0

    fn adjacency_matrix_flat(inout self, buf: UnsafePointer[Float64]):
        """Write dense (n×n) adjacency matrix into buf (must be n*n Float64).

        Uses zero-copy write into caller-provided buffer.
        """
        let n = self.n_vertices
        # Zero the buffer
        for i in range(n * n):
            buf.store(i, 0.0)

        for edge in self.edges:
            let idx = edge.source * n + edge.target
            buf.store(idx, buf.load(idx) + edge.weight)
            if not self.directed:
                let idx2 = edge.target * n + edge.source
                buf.store(idx2, buf.load(idx2) + edge.weight)

    fn degree(self, vertex: Int) -> Float64:
        """Sum of outgoing edge weights for a vertex."""
        var total: Float64 = 0.0
        let weights = self.adj_weights[vertex]
        for k in range(weights.size):
            total += weights[k]
        return total

    @staticmethod
    fn build_from_transitions(
        transitions: DynamicVector[Tuple[Int, Int]],
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
            if t.get[0]() > max_v:
                max_v = t.get[0]()
            if t.get[1]() > max_v:
                max_v = t.get[1]()

        # Add vertices 0..max_v
        for _ in range(max_v + 1):
            graph.add_vertex()

        # Count transitions into a simple map (src*max_n + tgt) → weight
        let n = max_v + 1
        var counts = DynamicVector[Float64]()
        for _ in range(n * n):
            counts.push_back(0.0)

        for t in transitions:
            let src = t.get[0]()
            let tgt = t.get[1]()
            let idx = src * n + tgt
            counts[idx] = counts[idx] + 1.0

        # Add edges with aggregated weights
        for src in range(n):
            for tgt in range(n):
                let w = counts[src * n + tgt]
                if w > 0.0:
                    graph.add_edge(src, tgt, w)

        return graph
