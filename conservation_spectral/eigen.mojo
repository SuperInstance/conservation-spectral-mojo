"""Eigendecomposition — QR algorithm for symmetric matrices.

Implements the symmetric QR algorithm with Wilkinson shifts for
eigendecomposition of Laplacian matrices. Pure Mojo, no Python dependencies.
"""

from collections.dynamic_vector import DynamicVector
from .laplacian import Laplacian


@value
struct EigenDecomposition:
    """Result of eigendecomposition of a Laplacian."""
    var n: Int
    var eigenvalues: UnsafePointer[Float64]    # (n,) sorted ascending
    var eigenvectors: UnsafePointer[Float64]   # (n, n) columns = eigenvectors
    var laplacian_type: StringLiteral
    var _owned: Bool

    fn __init__(
        inout self,
        n: Int,
        laplacian_type: StringLiteral = "symmetric_normalized",
    ):
        self.n = n
        self.laplacian_type = laplacian_type
        self._owned = True
        self.eigenvalues = UnsafePointer[Float64].alloc(n)
        self.eigenvectors = UnsafePointer[Float64].alloc(n * n)
        for i in range(n):
            self.eigenvalues.store(i, 0.0)
        for i in range(n * n):
            self.eigenvectors.store(i, 0.0)

    fn __del__(owned self):
        if self._owned:
            self.eigenvalues.free()
            self.eigenvectors.free()

    fn num_vectors(self) -> Int:
        return self.n

    fn num_vertices(self) -> Int:
        return self.n

    fn get_eigenvector(self, k: Int) -> UnsafePointer[Float64]:
        """Get pointer to the k-th eigenvector (column k)."""
        return self.eigenvectors + k * self.n


fn eigendecompose(
    owned lap: Laplacian,
    num_vectors: Int = 0,
    laplacian_type: StringLiteral = "symmetric_normalized",
) -> EigenDecomposition:
    """Compute full eigendecomposition of a Laplacian.

    Uses the symmetric QR algorithm with shifts for the n×n matrix.
    Returns all eigenvalues sorted ascending and corresponding eigenvectors.

    Args:
        lap: Laplacian from build_laplacian().
        num_vectors: Number of eigenvectors (0 = all).
        laplacian_type: Label for the type used.

    Returns:
        EigenDecomposition with eigenvalues sorted ascending.
    """
    let n = lap.n
    var eigen = EigenDecomposition(n, laplacian_type)

    # Copy Laplacian matrix into a working buffer A (n×n)
    var A = UnsafePointer[Float64].alloc(n * n)
    for i in range(n * n):
        A.store(i, lap.matrix.load(i))

    # Symmetrize to avoid numerical issues: A = (A + A^T) / 2
    for i in range(n):
        for j in range(i + 1, n):
            let avg = (A.load(i * n + j) + A.load(j * n + i)) / 2.0
            A.store(i * n + j, avg)
            A.store(j * n + i, avg)

    # Initialize eigenvector matrix as identity
    for i in range(n):
        for j in range(n):
            eigen.eigenvectors.store(i * n + j, 1.0 if i == j else 0.0)

    # --- Tridiagonalization via Householder reflections ---
    # Reduce A to tridiagonal form T = Q^T A Q
    var v = UnsafePointer[Float64].alloc(n)
    var p = UnsafePointer[Float64].alloc(n)

    for k in range(n - 2):
        # Extract column below diagonal
        var sigma: Float64 = 0.0
        for i in range(k + 2, n):
            sigma += A.load(i * n + k) * A.load(i * n + k)

        let alpha = A.load((k + 1) * n + k)
        let r = sqrt(alpha * alpha + sigma)
        if r < 1e-15:
            continue

        let sign = 1.0 if alpha >= 0.0 else -1.0
        let v0 = alpha + sign * r
        var v_norm_sq = v0 * v0
        v.store(k + 1, v0)
        for i in range(k + 2, n):
            let val = A.load(i * n + k)
            v.store(i, val)
            v_norm_sq += val * val

        if v_norm_sq < 1e-30:
            continue

        let inv_v_norm_sq = 1.0 / v_norm_sq

        # p = A * v * (2 / v_norm_sq)
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(k + 1, n):
                dot += A.load(i * n + j) * v.load(j)
            p.store(i, 2.0 * dot * inv_v_norm_sq)

        # beta = v^T * p / 2
        var beta: Float64 = 0.0
        for i in range(k + 1, n):
            beta += v.load(i) * p.load(i)
        beta /= 2.0

        # q = p - beta * v
        var q = UnsafePointer[Float64].alloc(n)
        for i in range(n):
            q.store(i, p.load(i) - beta * v.load(i))

        # A = A - v * q^T - q * v^T  (only update lower-right submatrix)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                let val = A.load(i * n + j) - v.load(i) * q.load(j) - q.load(i) * v.load(j)
                A.store(i * n + j, val)
                A.store(j * n + i, val)  # keep symmetric

        # Update eigenvectors: Q = Q * (I - 2 * v * v^T / v_norm_sq)
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(k + 1, n):
                dot += eigen.eigenvectors.load(i * n + j) * v.load(j)
            let coeff = 2.0 * dot * inv_v_norm_sq
            for j in range(k + 1, n):
                let old = eigen.eigenvectors.load(i * n + j)
                eigen.eigenvectors.store(i * n + j, old - coeff * v.load(j))

        q.free()

    v.free()
    p.free()

    # --- Symmetric QR algorithm on the tridiagonal matrix ---
    # Extract diagonal and sub-diagonal
    var d = UnsafePointer[Float64].alloc(n)     # diagonal
    var e = UnsafePointer[Float64].alloc(n - 1) # sub-diagonal

    for i in range(n):
        d.store(i, A.load(i * n + i))
    for i in range(n - 1):
        e.store(i, A.load((i + 1) * n + i))

    A.free()

    # QR iterations with Wilkinson shift
    let max_iter = 100 * n
    var iter = 0
    var m = n  # active size

    while m > 1 and iter < max_iter:
        iter += 1

        # Find the largest unreduced submatrix [l..m-1]
        var l = m - 1
        while l > 0:
            let off = abs(e.load(l - 1))
            let diag_sum = abs(d.load(l - 1)) + abs(d.load(l))
            if off <= 1e-14 * diag_sum:
                e.store(l - 1, 0.0)
                break
            l -= 1

        if l == m - 1:
            m -= 1
            continue

        # Wilkinson shift
        let dd = (d.load(m - 2) - d.load(m - 1)) / 2.0
        let ee = e.load(m - 2) * e.load(m - 2)
        let sign_dd = 1.0 if dd >= 0.0 else -1.0
        let mu = d.load(m - 1) - ee / (dd + sign_dd * sqrt(dd * dd + ee))

        var x = d.load(l) - mu
        var z = e.load(l)

        for k in range(l, m - 1):
            # Givens rotation to zero out z
            let r = sqrt(x * x + z * z)
            if r < 1e-30:
                continue
            let c = x / r
            let s = -z / r

            # Update tridiagonal
            if k > l:
                e.store(k - 1, r)

            let d_k = d.load(k)
            let d_k1 = d.load(k + 1)
            let e_k = e.load(k)

            let h = d_k1 - d_k + e_k * s * (2.0 * c * e_k / r + s)
            # Simpler update:
            let w = c * c * d_k + s * s * d_k1 - 2.0 * c * s * e_k
            let w1 = s * s * d_k + c * c * d_k1 + 2.0 * c * s * e_k

            d.store(k, w)
            d.store(k + 1, w1)

            if k < m - 2:
                let new_e = c * e.load(k + 1) + s * d.load(k + 1)  # approximate
                e.store(k, c * e_k + s * (d_k1 - d.load(k + 1)))
                e.store(k + 1, -s * e_k + c * e.load(k + 1))
                x = e.load(k)
                z = s * d.load(k + 1)

            # Update eigenvectors
            for i in range(n):
                let v1 = eigen.eigenvectors.load(i * n + k)
                let v2 = eigen.eigenvectors.load(i * n + k + 1)
                eigen.eigenvectors.store(i * n + k, c * v1 - s * v2)
                eigen.eigenvectors.store(i * n + k + 1, s * v1 + c * v2)

        # Check convergence
        if abs(e.load(m - 2)) < 1e-14 * (abs(d.load(m - 2)) + abs(d.load(m - 1))):
            e.store(m - 2, 0.0)
            m -= 1

    # Store eigenvalues from diagonal
    for i in range(n):
        eigen.eigenvalues.store(i, d.load(i))

    # Sort eigenvalues and eigenvectors ascending
    # Simple selection sort (fine for typical graph sizes)
    for i in range(n - 1):
        var min_idx = i
        for j in range(i + 1, n):
            if eigen.eigenvalues.load(j) < eigen.eigenvalues.load(min_idx):
                min_idx = j
        if min_idx != i:
            # Swap eigenvalues
            let tmp_eval = eigen.eigenvalues.load(i)
            eigen.eigenvalues.store(i, eigen.eigenvalues.load(min_idx))
            eigen.eigenvalues.store(min_idx, tmp_eval)
            # Swap eigenvector columns
            for row in range(n):
                let tmp = eigen.eigenvectors.load(row * n + i)
                eigen.eigenvectors.store(row * n + i, eigen.eigenvectors.load(row * n + min_idx))
                eigen.eigenvectors.store(row * n + min_idx, tmp)

    d.free()
    e.free()

    return eigen
