# USERMANUAL — conservation-spectral-mojo

> SIMD-accelerated spectral graph analysis (Laplacians, eigendecomposition,
> conservation ratios, Cheeger, spectral entropy, sliding-window tracker)
> in Mojo. Ported from `conservation-spectral-python`.

**On-box verdict (2026-10-01, Mojo 1.2.0.dev2026100105):**
**construction layer VERIFIED bit-exact vs numpy; eigen layer BOOKED-DEFECTIVE.**
Graph/Laplacian construction and the solver-independent conservation
invariants (`L·1 = 0`, symmetry, unit diagonal) match numpy to machine
epsilon on all 7 parity cases. The Householder+QR eigensolver returns wrong
eigenvalues (up to ~1e+94 absolute error, and all-NaN on some inputs) — four
semantic defects are diagnosed below and intentionally NOT fixed
(booked, originals preserved). Details: [Booked findings](#booked-findings).

## Quickstart

```bash
# Toolchain (no pixi manifest — mojo.modular is NOT a pixi.toml; pixi run fails here)
export MODULAR_HOME=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/share/max
export PATH=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/bin:$PATH

# tests (flags BEFORE filename; -D ASSERT=all required for asserts)
mojo run -D ASSERT=all -I . tests/test_main.mojo

# parity vs Python oracle (numpy ground truth)
python3 python/run_parity.py

# working example (receipt)
mojo run -D ASSERT=all -I . examples/demo.mojo

# benchmarks
mojo run -D ASSERT=all -I . benchmarks/bench.mojo
```

`-I .` is **required** on every mojo invocation: imports resolve from the
MAIN FILE's directory, and the `conservation_spectral` package lives at the
repo root. From any other directory, point `-I` at this repo root.

## What it is

Pipeline: `TensionGraph` → Laplacian (`unnormalized`, `symmetric_normalized`,
`random_walk_normalized`; scalar + SIMD builders) → eigendecomposition
→ conservation analysis:

- **Conservation ratio** CR(k): variance of the gradient of `attribute·φ_k`
  (how well an attribute is conserved along eigenmode k).
- **Spectral gap**: largest consecutive eigenvalue gap, skipping λ₀→λ₁.
- **Cheeger approx**: λ₁/2.
- **Spectral fingerprint**: entropy `H = -Σ p·ln p`, `p = |λ|/Σ|λ|`;
  effective dimension `e^H`.
- **Anomaly count**: z-score outliers (>2σ) over the first 3 eigenvectors.
- **Tracker**: sliding window of observations → argmax transition graph →
  baseline z-score alerting.

In this SDK the "conservation laws" are: `L·1 = 0` (a uniform attribute is
conserved — exact for unnormalized L, and for symmetric-normalized L when
all degrees are positive), symmetry of L where the variant promises it, and
unit diagonal for normalized variants on positive-degree vertices. All of
these **hold to machine epsilon** in the verified construction layer.

## Test + parity contract

| Command | Expected result on this box |
|---|---|
| `mojo run -D ASSERT=all -I . tests/test_main.mojo` | **10 passed, 1 failed** — the single failure is `Eigendecomposition`, which is the booked solver defect reporting itself. |
| `python3 python/run_parity.py` | Construction parity **PASS** (deg Δ=0, L Δ=0 on all 7 cases incl. degenerate); conservation invariants **PASS (machine epsilon)**; eigenvalue/metric parity **FAIL (booked)**. |
| `mojo run -D ASSERT=all -I . examples/demo.mojo` | Receipt: 3 invariant checks PASS, no spurious tracker alert, metrics section printed under BOOKED-FAIL banner. |

Parity cases: `ring8_unnorm`, `ring8_sym`, `path6_rw`,
`rand10_directed_sym` (bit-exact LCG replication of bench's random graph),
`star5_unnorm`, `two_isolated_sym` (L = I degenerate), `single_vertex`.

## Booked findings (semantic, NOT fixed — originals preserved)

All in `conservation_spectral/eigen.mojo`. Verified by numpy replication of
the exact algorithm (bit-for-bit agreement with the Mojo output at every
stage; evidence harness in `diagnostics/`):

1. **Householder β scaling** — `beta = vᵀp/2; A -= v·qᵀ + q·vᵀ` uses rank-2
   coefficient `2β = vᵀp`, but the correct coefficient is `2(vᵀp)/(vᵀv)`.
   Off by factor `vᵀv/2`; inflates the tridiagonal diagonal multiplicatively
   (P4: 2 → 10 → 138).
2. **Double-applied Householder update** — the trailing-block update loop
   visits both `(i,j)` and `(j,i)`, recomputing from already-mutated entries,
   so every off-diagonal pair is corrected twice (−1 → +1 → +3). Fix shape:
   loop upper triangle only, or compute from a snapshot.
3. **Missing explicit subdiagonal assignment** — after each reflection,
   `A[k+1][k] = -sign·r` is never written (loop scope excludes it), so the
   extracted T mixes pre- and post-reflection values.
4. **QR Givens rotation update is an approximation** — the code admits it in
   a dead comment (`# approximate`, plus unused `h`, `new_e` locals). Even
   fed a *perfect* tridiagonal (P4 Laplacian, true evals
   [0, 0.586, 2, 3.414]), it returns [-4.09, 0.118, 3.54, 6.43].

Consequences: every eigen-derived number (gap, Cheeger, entropy, effective
dimension, anomalies, eigenvalue fields of ratios) is untrustworthy for
n ≥ 3. The solver can also emit **all-NaN** eigenvalues (ring P12) and
values up to ~4e+94 (random n=10). Tests 7–11 pass only because they assert
weak non-negativity invariants that hold on garbage. Side effect: the QR
never converges, so `eigendecompose` always burns its full iteration budget
(~10 s at n=256).

Other booked (smaller) findings:
- `tracker.feed` leaks the previous `_current_ratios` buffer each call
  (original behavior, preserved).
- `__init__.mojo` originally imported a nonexistent `cheeger_constant`
  (fixed to `cheeger_constant_approx` — mechanical necessity).
- `tests/test_main.mojo.orig-20261001` — pre-fix archive; two tests built
  DIRECTED graphs while asserting UNDIRECTED-Laplacian constants
  (d=[1,2,1], λ₀≈0). Constants kept as spec, construction corrected.

## This box receipt (2026-10-01)

```
$ mojo run -D ASSERT=all -I . tests/test_main.mojo
[PASS] Graph basic construction
[PASS] Undirected graph
[PASS] Unnormalized Laplacian
[PASS] Normalized Laplacian
[PASS] SIMD Laplacian
[FAIL] Eigendecomposition        <- booked finding #1-4
[PASS] Conservation ratio
[PASS] Spectral gap
[PASS] Full analysis pipeline
[PASS] Conservation tracker
[PASS] Build from transitions
=== Results: 10 passed, 1 failed ===

$ python3 python/run_parity.py
case                     n    deg Δ      L Δ    eval Δ ...
ring8_unnorm             8       0        0       nan   CONSTRUCTION-PASS/EIGEN-FAIL
ring8_sym                8       0        0   4.73e+73 CONSTRUCTION-PASS/EIGEN-FAIL
path6_rw                 6       0        0      35.3  CONSTRUCTION-PASS/EIGEN-FAIL
rand10_directed_sym     10       0        0   4.03e+94 CONSTRUCTION-PASS/EIGEN-FAIL
star5_unnorm             5       0        0   2.94e+12 CONSTRUCTION-PASS/EIGEN-FAIL
two_isolated_sym         2       0        0        0   FULL-PASS
single_vertex            1       0        0        0   FULL-PASS
Construction parity: PASS · Conservation invariants: PASS (machine eps)
Eigenvalue/metric parity: FAIL — booked
```

## Troubleshooting (traps hit on this box, Mojo 1.2.0.dev2026100105)

**Toolchain**
- `pixi run mojo` fails here: `mojo.modular` is not a pixi manifest (pixi
  wants `pixi.toml`/`pyproject.toml`). Use the MODULAR_HOME+PATH exports
  above.
- Flags go BEFORE the filename: `mojo run -D ASSERT=all -I . path.mojo`.
  Flags after the filename are silently ignored. `-D ASSERT=all` is
  required for `assert` to fire at all.

**Language removals** (all hit here)
- `fn` removed → `def`. `let` removed → `var`. `owned`/`borrowed` param
  keywords removed (params are immutable borrows by default; move at the
  call site with `x^`).
- `@value` removed → plain `struct` + explicit `def __init__(out self, ...)`.
  Destructor is now `def __deinit__(deinit self):` (`__del__` deprecated).
- Module-level `var`/`alias` banned.
- `UnsafePointer` deprecated → `Pointer[T, MutUntrackedOrigin]` (struct
  fields/params need the explicit origin). Allocation is the free function
  `alloc[T](n)` — `Pointer[T].alloc` does not exist. `load/store/free/+` are
  deprecated spellings of `unsafe_load/unsafe_store/unsafe_free/unsafe_offset`.
  Pointer truthiness (`if ptr:`) is a compile error ("non-null by design") —
  gate on a length/flag instead; null-slot patterns become
  `alloc[Float64](1)` dummies + a Bool flag.
- `DynamicVector` → `List` (in prelude): `push_back`→`append`, `.size`→
  `len(x)`. `List` is NOT ImplicitlyCopyable: binding `var x = list[i]` for
  a List element is rejected (index directly); List-holding returns need
  `return x^`; `append` of an lvalue needs `^` (rvalues are fine).
- `StringRef` gone → `String`. `StringLiteral` is not concrete as a struct
  field type — use `String`.
- `raise("msg")` → `raise Error("msg")`, and the enclosing `def` needs
  `raises`.
- `t.get[i]()` → `t[i]`. `from collections import ...` does not exist;
  `from collections.dynamic_vector import` neither — List is prelude.
- `math.sqrt` etc: `from std.math import sqrt, log, exp, sin, cos`;
  `now()` dead → `from std import time` + `time.perf_counter_ns()`.
- Prelude quirk: `Pointer`/`alloc` only resolve if the file has ≥1
  `from std import ...` line (any std import).
- Scalar `min`/`max` survive; **SIMD** comparisons reduce to scalar Bool and
  there is no elementwise max/min (free function or method). `abs` is prelude.
- Intra-package imports: relative `from .graph import ...` did not resolve —
  use absolute `from conservation_spectral.graph import ...`, with `-I .`
  and `__init__.mojo` present.
- `String(x, 3)` and `.align_right()` do not exist; `len()` on a String is a
  compile error (use `.byte_length()`). Round manually:
  `String(Float64(Int(x * 1000)) / 1000)`.
- Nested functions cannot mutate captured locals (and `nonterminal` is not a
  keyword) — the test harness uses a `Counters` struct passed as `mut self`.

**Repo-specific**
- Eigensolver wrong/NaN → booked, see [Booked findings](#booked-findings).
  Do not trust gap/Cheeger/entropy/anomaly numbers for n ≥ 3.
- `mojo build -I . tests/test_main.mojo -o <bin>` works for a compiled
  binary; run scripts need the same `-I .`.
- Diagnostic evidence lives in `diagnostics/` (Householder stage isolation,
  corrected-beta proof, parity dump) — kept, never deleted.
