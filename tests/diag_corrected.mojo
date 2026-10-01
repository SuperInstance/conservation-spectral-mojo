from std import time
from std.math import sqrt

# DIAG (booked-finding evidence, NOT a repo fix): eigen.mojo pipeline with the
# single corrected line (beta scaling) to prove the QR stage is sound.
def run(L_in: Pointer[Float64, MutUntrackedOrigin], n: Int, evals_out: Pointer[Float64, MutUntrackedOrigin]):
    var A = alloc[Float64](n * n)
    for i in range(n * n):
        A.unsafe_store(i, L_in.unsafe_load(i))
    for i in range(n):
        for j in range(i + 1, n):
            var avg = (A.unsafe_load(i * n + j) + A.unsafe_load(j * n + i)) / 2.0
            A.unsafe_store(i * n + j, avg)
            A.unsafe_store(j * n + i, avg)

    var v = alloc[Float64](n)
    var p = alloc[Float64](n)
    for k in range(n - 2):
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
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(k + 1, n):
                dot += A.unsafe_load(i * n + j) * v.unsafe_load(j)
            p.unsafe_store(i, 2.0 * dot * inv_v_norm_sq)
        var beta: Float64 = 0.0
        for i in range(k + 1, n):
            beta += v.unsafe_load(i) * p.unsafe_load(i)
        # CORRECTED LINE (vs repo): beta must be scaled by vᵀp/(vᵀv), not vᵀp/2
        beta = beta * inv_v_norm_sq  # i.e. beta = (vᵀp/2) / (vᵀv/2) = vᵀp/(vᵀv)
        var q = alloc[Float64](n)
        for i in range(n):
            q.unsafe_store(i, p.unsafe_load(i) - beta * v.unsafe_load(i))
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                var val = A.unsafe_load(i * n + j) - v.unsafe_load(i) * q.unsafe_load(j) - q.unsafe_load(i) * v.unsafe_load(j)
                A.unsafe_store(i * n + j, val)
                A.unsafe_store(j * n + i, val)
        q.unsafe_free()

    var d = alloc[Float64](n)
    var e = alloc[Float64](n - 1 if n > 0 else 1)
    for i in range(n):
        d.unsafe_store(i, A.unsafe_load(i * n + i))
    for i in range(n - 1):
        e.unsafe_store(i, A.unsafe_load((i + 1) * n + i))
    print("T diag (corrected):", d.unsafe_load(0), d.unsafe_load(1), d.unsafe_load(2), d.unsafe_load(3))
    print("T subdiag (corrected):", e.unsafe_load(0), e.unsafe_load(1), e.unsafe_load(2))

    # original QR stage verbatim from eigen.mojo
    var max_iter = 100 * n
    var n_iter = 0
    var m = n
    while m > 1 and n_iter < max_iter:
        n_iter += 1
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
        var dd = (d.unsafe_load(m - 2) - d.unsafe_load(m - 1)) / 2.0
        var ee = e.unsafe_load(m - 2) * e.unsafe_load(m - 2)
        var sign_dd = 1.0 if dd >= 0.0 else -1.0
        var mu = d.unsafe_load(m - 1) - ee / (dd + sign_dd * sqrt(dd * dd + ee))
        var x = d.unsafe_load(l) - mu
        var z = e.unsafe_load(l)
        for k in range(l, m - 1):
            var r = sqrt(x * x + z * z)
            if r < 1e-30:
                continue
            var c = x / r
            var s = -z / r
            if k > l:
                e.unsafe_store(k - 1, r)
            var d_k = d.unsafe_load(k)
            var d_k1 = d.unsafe_load(k + 1)
            var e_k = e.unsafe_load(k)
            var w = c * c * d_k + s * s * d_k1 - 2.0 * c * s * e_k
            var w1 = s * s * d_k + c * c * d_k1 + 2.0 * c * s * e_k
            d.unsafe_store(k, w)
            d.unsafe_store(k + 1, w1)
            if k < m - 2:
                e.unsafe_store(k, c * e_k + s * (d_k1 - w1))
                e.unsafe_store(k + 1, -s * e_k + c * e.unsafe_load(k + 1))
                x = e.unsafe_load(k)
                z = s * w1
        if abs(e.unsafe_load(m - 2)) < 1e-14 * (abs(d.unsafe_load(m - 2)) + abs(d.unsafe_load(m - 1))):
            e.unsafe_store(m - 2, 0.0)
            m -= 1
    for i in range(n):
        evals_out.unsafe_store(i, d.unsafe_load(i))

def main():
    var n = 4
    var L = alloc[Float64](n * n)
    var rows = List[List[Float64]]()
    var r0 = List[Float64]()
    r0.append(1.0); r0.append(-1.0); r0.append(0.0); r0.append(0.0)
    var r1 = List[Float64]()
    r1.append(-1.0); r1.append(2.0); r1.append(-1.0); r1.append(0.0)
    var r2 = List[Float64]()
    r2.append(0.0); r2.append(-1.0); r2.append(2.0); r2.append(-1.0)
    var r3 = List[Float64]()
    r3.append(0.0); r3.append(0.0); r3.append(-1.0); r3.append(1.0)
    rows.append(r0^); rows.append(r1^); rows.append(r2^); rows.append(r3^)
    for i in range(n):
        for j in range(n):
            L.unsafe_store(i * n + j, rows[i][j])
    var evals = alloc[Float64](n)
    run(L, n, evals)
    print("evals (corrected Householder + repo QR):", evals.unsafe_load(0), evals.unsafe_load(1), evals.unsafe_load(2), evals.unsafe_load(3))
