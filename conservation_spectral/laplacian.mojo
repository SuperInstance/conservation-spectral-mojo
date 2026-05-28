"""Laplacian construction — SIMD-optimized normalized and unnormalized variants.

Uses SIMD[DType.float64, 4] for vectorized matrix operations and
UnsafePointer for zero-copy matrix access.
"""

from collections.dynamic_vector import DynamicVector
from .graph import TensionGraph


@value
struct Laplacian:
    """Computed Laplacian from a TensionGraph.

    Stores the matrix as a flat n×n array behind an UnsafePointer
    for zero-copy, SIMD-friendly access.
    """
    var n: Int
    var matrix: UnsafePointer[Float64]   # flat n×n Laplacian
    var weight_matrix: UnsafePointer[Float64]  # flat n×n adjacency weights
    var degree_vec: UnsafePointer[Float64]     # degree vector (diagonal of D)
    var is_normalized: Bool
    var laplacian_type: StringLiteral
    var _owned: Bool  # whether we own the memory (for destructor)

    fn __init__(
        inout self,
        n: Int,
        laplacian_type: StringLiteral = "symmetric_normalized",
    ):
        self.n = n
        self.laplacian_type = laplacian_type
        self.is_normalized = (laplacian_type != "unnormalized")
        self._owned = True
        self.matrix = UnsafePointer[Float64].alloc(n * n)
        self.weight_matrix = UnsafePointer[Float64].alloc(n * n)
        self.degree_vec = UnsafePointer[Float64].alloc(n)
        # Zero-initialize
        for i in range(n * n):
            self.matrix.store(i, 0.0)
            self.weight_matrix.store(i, 0.0)
        for i in range(n):
            self.degree_vec.store(i, 0.0)

    fn __del__(owned self):
        if self._owned:
            self.matrix.free()
            self.weight_matrix.free()
            self.degree_vec.free()

    fn load_element(self, mat: UnsafePointer[Float64], row: Int, col: Int) -> Float64:
        return mat.load(row * self.n + col)

    fn store_element(self, mat: UnsafePointer[Float64], row: Int, col: Int, val: Float64):
        mat.store(row * self.n + col, val)


fn build_laplacian(
    owned graph: TensionGraph,
    laplacian_type: StringLiteral = "symmetric_normalized",
) -> Laplacian:
    """Build a Laplacian from a TensionGraph.

    Constructs the weight matrix, degree vector, and Laplacian using
    SIMD-accelerated operations.

    Supports:
        - "unnormalized": L = D - W
        - "symmetric_normalized": L = I - D^{-1/2} W D^{-1/2}
        - "random_walk_normalized": L = I - D^{-1} W
    """
    let n = graph.n_vertices
    var lap = Laplacian(n, laplacian_type)

    # Build weight matrix from graph edges
    graph.adjacency_matrix_flat(lap.weight_matrix)

    # Compute degree vector
    for i in range(n):
        var deg: Float64 = 0.0
        for j in range(n):
            deg += lap.load_element(lap.weight_matrix, i, j)
        lap.degree_vec.store(i, deg)

    # Build Laplacian based on type
    if laplacian_type == "unnormalized":
        # L = D - W
        for i in range(n):
            for j in range(n):
                if i == j:
                    lap.store_element(lap.matrix, i, j,
                        lap.degree_vec.load(i) - lap.load_element(lap.weight_matrix, i, j))
                else:
                    lap.store_element(lap.matrix, i, j,
                        -lap.load_element(lap.weight_matrix, i, j))

    elif laplacian_type == "symmetric_normalized":
        # L = I - D^{-1/2} W D^{-1/2}
        # Precompute D^{-1/2}
        var d_inv_sqrt = UnsafePointer[Float64].alloc(n)
        for i in range(n):
            let d = lap.degree_vec.load(i)
            d_inv_sqrt.store(i, 1.0 / sqrt(d) if d > 0.0 else 0.0)

        for i in range(n):
            for j in range(n):
                let w = lap.load_element(lap.weight_matrix, i, j)
                let val = -d_inv_sqrt.load(i) * w * d_inv_sqrt.load(j)
                if i == j:
                    lap.store_element(lap.matrix, i, j, 1.0 + val)
                else:
                    lap.store_element(lap.matrix, i, j, val)

        d_inv_sqrt.free()

    elif laplacian_type == "random_walk_normalized":
        # L = I - D^{-1} W
        for i in range(n):
            let d = lap.degree_vec.load(i)
            let d_inv = 1.0 / d if d > 0.0 else 0.0
            for j in range(n):
                let w = lap.load_element(lap.weight_matrix, i, j)
                if i == j:
                    lap.store_element(lap.matrix, i, j, 1.0 - d_inv * w)
                else:
                    lap.store_element(lap.matrix, i, j, -d_inv * w)

    return lap


fn build_laplacian_simd(
    owned graph: TensionGraph,
    laplacian_type: StringLiteral = "symmetric_normalized",
) -> Laplacian:
    """Build a Laplacian using SIMD[DType.float64, 4] vectorized operations.

    Processes 4 matrix entries at a time for the inner loop, yielding
    ~4× throughput on the matrix construction phase.
    """
    let n = graph.n_vertices
    let simd_width: Int = 4  # SIMD[DType.float64, 4]
    var lap = Laplacian(n, laplacian_type)

    # Build weight matrix
    graph.adjacency_matrix_flat(lap.weight_matrix)

    # Compute degree vector with SIMD accumulation
    for i in range(n):
        var deg = SIMD[DType.float64, 4](0.0)
        var scalar_deg: Float64 = 0.0
        let row_offset = i * n
        var j: Int = 0
        # SIMD loop: process 4 entries at a time
        while j + simd_width <= n:
            let w = SIMD[DType.float64, 4](
                lap.weight_matrix.load(row_offset + j),
                lap.weight_matrix.load(row_offset + j + 1),
                lap.weight_matrix.load(row_offset + j + 2),
                lap.weight_matrix.load(row_offset + j + 3),
            )
            deg = deg + w
            j += simd_width
        # Horizontal reduction
        scalar_deg = deg[0] + deg[1] + deg[2] + deg[3]
        # Scalar tail
        while j < n:
            scalar_deg += lap.weight_matrix.load(row_offset + j)
            j += 1
        lap.degree_vec.store(i, scalar_deg)

    # Build Laplacian matrix with SIMD
    if laplacian_type == "unnormalized":
        for i in range(n):
            let d = lap.degree_vec.load(i)
            let row_offset = i * n
            var j: Int = 0
            while j + simd_width <= n:
                # Build diagonal mask: 1.0 where i==j, 0.0 elsewhere
                let diag_mask = SIMD[DType.float64, 4](
                    1.0 if (j == i) else 0.0,
                    1.0 if (j + 1 == i) else 0.0,
                    1.0 if (j + 2 == i) else 0.0,
                    1.0 if (j + 3 == i) else 0.0,
                )
                let w = SIMD[DType.float64, 4](
                    lap.weight_matrix.load(row_offset + j),
                    lap.weight_matrix.load(row_offset + j + 1),
                    lap.weight_matrix.load(row_offset + j + 2),
                    lap.weight_matrix.load(row_offset + j + 3),
                )
                # L[i,j] = D[i]*diag_mask - W[i,j]
                let l_val = d * diag_mask - w
                lap.matrix.store(row_offset + j, l_val[0])
                lap.matrix.store(row_offset + j + 1, l_val[1])
                lap.matrix.store(row_offset + j + 2, l_val[2])
                lap.matrix.store(row_offset + j + 3, l_val[3])
                j += simd_width
            # Scalar tail
            while j < n:
                let val = d - lap.weight_matrix.load(row_offset + j) if j == i else -lap.weight_matrix.load(row_offset + j)
                lap.matrix.store(row_offset + j, val)
                j += 1

    elif laplacian_type == "symmetric_normalized":
        # L = I - D^{-1/2} W D^{-1/2}
        var d_inv_sqrt = UnsafePointer[Float64].alloc(n)
        for i in range(n):
            let d = lap.degree_vec.load(i)
            d_inv_sqrt.store(i, 1.0 / sqrt(d) if d > 0.0 else 0.0)

        for i in range(n):
            let di = d_inv_sqrt.load(i)
            let row_offset = i * n
            var j: Int = 0
            while j + simd_width <= n:
                let diag_mask = SIMD[DType.float64, 4](
                    1.0 if (j == i) else 0.0,
                    1.0 if (j + 1 == i) else 0.0,
                    1.0 if (j + 2 == i) else 0.0,
                    1.0 if (j + 3 == i) else 0.0,
                )
                let dj = SIMD[DType.float64, 4](
                    d_inv_sqrt.load(j),
                    d_inv_sqrt.load(j + 1),
                    d_inv_sqrt.load(j + 2),
                    d_inv_sqrt.load(j + 3),
                )
                let w = SIMD[DType.float64, 4](
                    lap.weight_matrix.load(row_offset + j),
                    lap.weight_matrix.load(row_offset + j + 1),
                    lap.weight_matrix.load(row_offset + j + 2),
                    lap.weight_matrix.load(row_offset + j + 3),
                )
                # L[i,j] = diag_mask - di * W[i,j] * dj
                let l_val = diag_mask - di * w * dj
                lap.matrix.store(row_offset + j, l_val[0])
                lap.matrix.store(row_offset + j + 1, l_val[1])
                lap.matrix.store(row_offset + j + 2, l_val[2])
                lap.matrix.store(row_offset + j + 3, l_val[3])
                j += simd_width
            while j < n:
                let val = 1.0 - di * lap.weight_matrix.load(row_offset + j) * d_inv_sqrt.load(j) if j == i \
                    else -di * lap.weight_matrix.load(row_offset + j) * d_inv_sqrt.load(j)
                lap.matrix.store(row_offset + j, val)
                j += 1

        d_inv_sqrt.free()

    elif laplacian_type == "random_walk_normalized":
        for i in range(n):
            let d = lap.degree_vec.load(i)
            let d_inv = 1.0 / d if d > 0.0 else 0.0
            let row_offset = i * n
            var j: Int = 0
            while j + simd_width <= n:
                let diag_mask = SIMD[DType.float64, 4](
                    1.0 if (j == i) else 0.0,
                    1.0 if (j + 1 == i) else 0.0,
                    1.0 if (j + 2 == i) else 0.0,
                    1.0 if (j + 3 == i) else 0.0,
                )
                let w = SIMD[DType.float64, 4](
                    lap.weight_matrix.load(row_offset + j),
                    lap.weight_matrix.load(row_offset + j + 1),
                    lap.weight_matrix.load(row_offset + j + 2),
                    lap.weight_matrix.load(row_offset + j + 3),
                )
                let l_val = diag_mask - d_inv * w
                lap.matrix.store(row_offset + j, l_val[0])
                lap.matrix.store(row_offset + j + 1, l_val[1])
                lap.matrix.store(row_offset + j + 2, l_val[2])
                lap.matrix.store(row_offset + j + 3, l_val[3])
                j += simd_width
            while j < n:
                let val = 1.0 - d_inv * lap.weight_matrix.load(row_offset + j) if j == i \
                    else -d_inv * lap.weight_matrix.load(row_offset + j)
                lap.matrix.store(row_offset + j, val)
                j += 1

    return lap
