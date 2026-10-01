from std import time
from std.math import sqrt

# Copy of eigen.mojo's Householder stage, isolated: prints tridiagonal T
def householder_stage(A_in: Pointer[Float64, MutUntrackedOrigin], n: Int, d: Pointer[Float64, MutUntrackedOrigin], e: Pointer[Float64, MutUntrackedOrigin]):
    var A = alloc[Float64](n * n)
    for i in range(n * n):
        A.unsafe_store(i, A_in.unsafe_load(i))
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
        beta /= 2.0
        var q = alloc[Float64](n)
        for i in range(n):
            q.unsafe_store(i, p.unsafe_load(i) - beta * v.unsafe_load(i))
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                var val = A.unsafe_load(i * n + j) - v.unsafe_load(i) * q.unsafe_load(j) - q.unsafe_load(i) * v.unsafe_load(j)
                A.unsafe_store(i * n + j, val)
                A.unsafe_store(j * n + i, val)
        q.unsafe_free()
    for i in range(n):
        d.unsafe_store(i, A.unsafe_load(i * n + i))
    for i in range(n - 1):
        e.unsafe_store(i, A.unsafe_load((i + 1) * n + i))
    A.unsafe_free()
    v.unsafe_free()
    p.unsafe_free()

def main():
    var n = 4
    var L = alloc[Float64](n * n)
    # undirected path P4 unnormalized Laplacian
    var r0 = List[Float64]()
    r0.append(1.0); r0.append(-1.0); r0.append(0.0); r0.append(0.0)
    var r1 = List[Float64]()
    r1.append(-1.0); r1.append(2.0); r1.append(-1.0); r1.append(0.0)
    var r2 = List[Float64]()
    r2.append(0.0); r2.append(-1.0); r2.append(2.0); r2.append(-1.0)
    var r3 = List[Float64]()
    r3.append(0.0); r3.append(0.0); r3.append(-1.0); r3.append(1.0)
    var rows = List[List[Float64]]()
    rows.append(r0^); rows.append(r1^); rows.append(r2^); rows.append(r3^)
    for i in range(n):
        for j in range(n):
            L.unsafe_store(i * n + j, rows[i][j])
    var d = alloc[Float64](n)
    var e = alloc[Float64](n)
    householder_stage(L, n, d, e)
    print("T diag:", d.unsafe_load(0), d.unsafe_load(1), d.unsafe_load(2), d.unsafe_load(3))
    print("T subdiag:", e.unsafe_load(0), e.unsafe_load(1), e.unsafe_load(2))
