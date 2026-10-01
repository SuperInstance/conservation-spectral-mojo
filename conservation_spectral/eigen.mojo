"""Eigendecomposition — QR algorithm for symmetric matrices.

Implements the symmetric QR algorithm with Wilkinson shifts for
eigendecomposition of Laplacian matrices. Pure Mojo, no Python dependencies.

Mojo port (verified on-box 2026-10-01): Pointer[Float64, MutUntrackedOrigin],
alloc[Float64](n), __deinit__(deinit self).
"""

from std import time
from std.math import sqrt

from conservation_spectral.laplacian import Laplacian


struct EigenDecomposition:
    """Result of eigendecomposition of a Laplacian."""
    var n: Int
    var eigenvalues: Pointer[Float64, MutUntrackedOrigin]    # (n,) sorted ascending
    var eigenvectors: Pointer[Float64, MutUntrackedOrigin]   # (n, n) columns = eigenvectors
    var laplacian_type: String
    var _owned: Bool

    def __init__(
        out self,
        n: Int,
        laplacian_type: StringLiteral = "symmetric_normalized",
    ):
        self.n = n
        self.laplacian_type = laplacian_type
        self._owned = True
        self.eigenvalues = alloc[Float64](n)
        self.eigenvectors = alloc[Float64](n * n)
        for i in range(n):
            self.eigenvalues.unsafe_store(i, 0.0)
        for i in range(n * n):
            self.eigenvectors.unsafe_store(i, 0.0)

    def __deinit__(deinit self):
        if self._owned:
            self.eigenvalues.unsafe_free()
            self.eigenvectors.unsafe_free()

    def num_vectors(self) -> Int:
        return self.n

    def num_vertices(self) -> Int:
        return self.n

    def get_eigenvector(self, k: Int) -> Pointer[Float64, MutUntrackedOrigin]:
        """Get pointer to the k-th eigenvector (column k)."""
        return self.eigenvectors.unsafe_offset(k * self.n)


def eigendecompose(
    lap: Laplacian,
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
    var n = lap.n
    var eigen = EigenDecomposition(n, laplacian_type)

    # Copy Laplacian matrix into a working buffer A (n×n)
    var A = alloc[Float64](n * n)
    for i in range(n * n):
        A.unsafe_store(i, lap.matrix.unsafe_load(i))

    # Symmetrize to avoid numerical issues: A = (A + A^T) / 2
    for i in range(n):
        for j in range(i + 1, n):
            var avg = (A.unsafe_load(i * n + j) + A.unsafe_load(j * n + i)) / 2.0
            A.unsafe_store(i * n + j, avg)
            A.unsafe_store(j * n + i, avg)

    # Initialize eigenvector matrix as identity
    for i in range(n):
        for j in range(n):
            eigen.eigenvectors.unsafe_store(i * n + j, 1.0 if i == j else 0.0)

    # --- Tridiagonalization via Householder reflections ---
    # Reduce A to tridiagonal form T = Q^T A Q
    var v = alloc[Float64](n)
    var p = alloc[Float64](n)

    for k in range(n - 2):
        # Extract column below diagonal
        var sigma: Float64 = 0.0
        for i in range(k + 2, n):
            sigma += A.unsafe_load(i * n + k) * A.unsafe_load(i * n + k)

        var alpha = A.unsafe_load((k + 1) * n + k)
        var r = sqrt(alpha * alpha + sigma)
        if r < 1e-15:
            continue

        var sign = 1.0 if alpha >= 0.0 else -1.0
        var v0 = alpha + sign * r
        var v_norm_sq = v0 * v0
        v.unsafe_store(k + 1, v0)
        for i in range(k + 2, n):
            var val = A.unsafe_load(i * n + k)
            v.unsafe_store(i, val)
            v_norm_sq += val * val

        if v_norm_sq < 1e-30:
            continue

        var inv_v_norm_sq = 1.0 / v_norm_sq

        # p = A * v * (2 / v_norm_sq)
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(k + 1, n):
                dot += A.unsafe_load(i * n + j) * v.unsafe_load(j)
            p.unsafe_store(i, 2.0 * dot * inv_v_norm_sq)

        # beta = v^T * p / 2
        var beta: Float64 = 0.0
        for i in range(k + 1, n):
            beta += v.unsafe_load(i) * p.unsafe_load(i)
        beta /= 2.0

        # q = p - beta * v
        var q = alloc[Float64](n)
        for i in range(n):
            q.unsafe_store(i, p.unsafe_load(i) - beta * v.unsafe_load(i))

        # A = A - v * q^T - q * v^T  (only update lower-right submatrix)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                var val = A.unsafe_load(i * n + j) - v.unsafe_load(i) * q.unsafe_load(j) - q.unsafe_load(i) * v.unsafe_load(j)
                A.unsafe_store(i * n + j, val)
                A.unsafe_store(j * n + i, val)  # keep symmetric

        # Update eigenvectors: Q = Q * (I - 2 * v * v^T / v_norm_sq)
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(k + 1, n):
                dot += eigen.eigenvectors.unsafe_load(i * n + j) * v.unsafe_load(j)
            var coeff = 2.0 * dot * inv_v_norm_sq
            for j in range(k + 1, n):
                var old = eigen.eigenvectors.unsafe_load(i * n + j)
                eigen.eigenvectors.unsafe_store(i * n + j, old - coeff * v.unsafe_load(j))

        q.unsafe_free()

    v.unsafe_free()
    p.unsafe_free()

    # --- Symmetric QR algorithm on the tridiagonal matrix ---
    # Extract diagonal and sub-diagonal
    var d = alloc[Float64](n)     # diagonal
    var e = alloc[Float64](n - 1 if n > 0 else 1)  # sub-diagonal

    for i in range(n):
        d.unsafe_store(i, A.unsafe_load(i * n + i))
    for i in range(n - 1):
        e.unsafe_store(i, A.unsafe_load((i + 1) * n + i))

    A.unsafe_free()

    # QR iterations with Wilkinson shift
    var max_iter = 100 * n
    var n_iter = 0
    var m = n  # active size

    while m > 1 and n_iter < max_iter:
        n_iter += 1

        # Find the largest unreduced submatrix [l..m-1]
        var l = m - 1
        while l > 0:
            var off = abs(e.unsafe_load(l - 1))
            var diag_sum = abs(d.unsafe_load(l - 1)) + abs(d.unsafe_load(l))
            if off <= 1e-14 * diag_sum:
                e.unsafe_store(l - 1, 0.0)
                break
            l -= 1

        if l == m - 1:
            m -= 1
            continue

        # Wilkinson shift
        var dd = (d.unsafe_load(m - 2) - d.unsafe_load(m - 1)) / 2.0
        var ee = e.unsafe_load(m - 2) * e.unsafe_load(m - 2)
        var sign_dd = 1.0 if dd >= 0.0 else -1.0
        var mu = d.unsafe_load(m - 1) - ee / (dd + sign_dd * sqrt(dd * dd + ee))

        var x = d.unsafe_load(l) - mu
        var z = e.unsafe_load(l)

        for k in range(l, m - 1):
            # Givens rotation to zero out z
            var r = sqrt(x * x + z * z)
            if r < 1e-30:
                continue
            var c = x / r
            var s = -z / r

            # Update tridiagonal
            if k > l:
                e.unsafe_store(k - 1, r)

            var d_k = d.unsafe_load(k)
            var d_k1 = d.unsafe_load(k + 1)
            var e_k = e.unsafe_load(k)

            # Rotation update of d[k], d[k+1]
            var w = c * c * d_k + s * s * d_k1 - 2.0 * c * s * e_k
            var w1 = s * s * d_k + c * c * d_k1 + 2.0 * c * s * e_k

            d.unsafe_store(k, w)
            d.unsafe_store(k + 1, w1)

            if k < m - 2:
                e.unsafe_store(k, c * e_k + s * (d_k1 - w1))
                e.unsafe_store(k + 1, -s * e_k + c * e.unsafe_load(k + 1))
                x = e.unsafe_load(k)
                z = s * w1

            # Update eigenvectors
            for i in range(n):
                var v1 = eigen.eigenvectors.unsafe_load(i * n + k)
                var v2 = eigen.eigenvectors.unsafe_load(i * n + k + 1)
                eigen.eigenvectors.unsafe_store(i * n + k, c * v1 - s * v2)
                eigen.eigenvectors.unsafe_store(i * n + k + 1, s * v1 + c * v2)

        # Check convergence
        if abs(e.unsafe_load(m - 2)) < 1e-14 * (abs(d.unsafe_load(m - 2)) + abs(d.unsafe_load(m - 1))):
            e.unsafe_store(m - 2, 0.0)
            m -= 1

    # Store eigenvalues from diagonal
    for i in range(n):
        eigen.eigenvalues.unsafe_store(i, d.unsafe_load(i))

    # Sort eigenvalues and eigenvectors ascending
    # Simple selection sort (fine for typical graph sizes)
    for i in range(n - 1):
        var min_idx = i
        for j in range(i + 1, n):
            if eigen.eigenvalues.unsafe_load(j) < eigen.eigenvalues.unsafe_load(min_idx):
                min_idx = j
        if min_idx != i:
            # Swap eigenvalues
            var tmp_eval = eigen.eigenvalues.unsafe_load(i)
            eigen.eigenvalues.unsafe_store(i, eigen.eigenvalues.unsafe_load(min_idx))
            eigen.eigenvalues.unsafe_store(min_idx, tmp_eval)
            # Swap eigenvector columns
            for row in range(n):
                var tmp = eigen.eigenvectors.unsafe_load(row * n + i)
                eigen.eigenvectors.unsafe_store(row * n + i, eigen.eigenvectors.unsafe_load(row * n + min_idx))
                eigen.eigenvectors.unsafe_store(row * n + min_idx, tmp)

    d.unsafe_free()
    e.unsafe_free()

    return eigen^
