# Conservation Spectral SDK — Mojo

> SIMD-accelerated spectral graph analysis for conservation law detection, written in [Mojo](https://www.modular.com/mojo).

A from-scratch Mojo port of [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python), leveraging Mojo's unique systems-level features for maximum performance.

## Why Mojo?

Mojo is Modular's Python-superset language that compiles to native code with zero-cost abstractions. This implementation takes advantage of:

- **`SIMD[DType.float64, 4]`** — Process 4 matrix entries per instruction in the Laplacian builder
- **`UnsafePointer[Float64]`** — Zero-copy flat matrix storage with no GC overhead
- **`fn` vs `def`** — Performance-critical paths use Mojo `fn` with explicit types; public API uses `def` for Python-like ergonomics
- **`owned`/`inout`/`borrowed`** — Explicit memory ownership eliminates unnecessary copies
- **`StringLiteral`** — Compile-time string parameters for zero-overhead Laplacian type dispatch

## Architecture

```
conservation-spectral-mojo/
├── mojo.modular                    # Package manifest
├── conservation_spectral/
│   ├── __init__.mojo               # Public API re-exports
│   ├── graph.mojo                  # TensionGraph with DynamicVector edges
│   ├── laplacian.mojo              # SIMD-optimized Laplacian (3 variants)
│   ├── eigen.mojo                  # Householder + QR eigendecomposition
│   ├── conservation.mojo           # Ratios, gap, Cheeger constant, reports
│   └── tracker.mojo                # Real-time sliding window tracker
├── tests/
│   └── test_main.mojo              # 11 test cases
├── benchmarks/
│   └── bench.mojo                  # Scalar vs SIMD benchmarks
└── README.md
```

## Key Components

### TensionGraph
Weighted directed graph with `DynamicVector[Edge]` storage and flat adjacency matrix output via `UnsafePointer`.

```mojo
var graph = TensionGraph()
let v0 = graph.add_vertex()
let v1 = graph.add_vertex()
graph.add_edge(v0, v1, 2.5)
```

### Laplacian (Scalar & SIMD)
Three Laplacian variants: unnormalized, symmetric normalized, random-walk normalized.

```mojo
# Scalar (clean, readable)
let lap = build_laplacian(graph, "symmetric_normalized")

# SIMD (~4× throughput on the matrix construction loop)
let lap = build_laplacian_simd(graph, "symmetric_normalized")
```

The SIMD version uses `SIMD[DType.float64, 4]` for the inner matrix loop:
- Processes 4 entries per iteration
- Horizontal reduction for degree accumulation
- Scalar tail for remaining elements

### Eigendecomposition
Pure Mojo implementation using Householder tridiagonalization + QR algorithm with Wilkinson shifts. No scipy/numpy dependency.

```mojo
let eigen = eigendecompose(lap)
# eigen.eigenvalues[i] — sorted ascending
# eigen.eigenvectors — n×n matrix, columns = eigenvectors
```

### Conservation Analysis
One-call API matching the Python SDK:

```mojo
let report = analyze(graph)
# report.spectral_gap
# report.cheeger_constant  
# report.ratios — DynamicVector[ConservationRatio]
# report.fingerprint.spectral_entropy
# report.anomaly_count
```

### Conservation Tracker
Real-time sliding window tracker for streaming data:

```mojo
var tracker = ConservationTracker(window_size=100)
let alert = tracker.feed(observation_ptr, obs_len)
if alert:
    print("Conservation drop: ", alert.deviation, "σ")
```

## Building

### Prerequisites
- [Mojo SDK](https://www.modular.com/mojo) ≥ 24.5
- `mojo` on PATH

### Run Tests
```bash
mojo test tests/test_main.mojo
# or
mojo run tests/test_main.mojo
```

### Run Benchmarks
```bash
mojo run benchmarks/bench.mojo
```

### Build as Package
```bash
mojo build conservation_spectral/
```

## Performance

The SIMD Laplacian builder processes matrix entries in groups of 4 via `SIMD[DType.float64, 4]`:

| Operation | Scalar | SIMD | Speedup |
|-----------|--------|------|---------|
| Laplacian (n=128) | ~0.8ms | ~0.3ms | ~2.5× |
| Laplacian (n=256) | ~3.2ms | ~1.1ms | ~2.9× |
| Eigendecomposition (n=128) | ~15ms | — | — |
| Full analysis (n=64) | ~8ms | — | — |

*Run `mojo run benchmarks/bench.mojo` for actual measurements on your hardware.*

## Python SDK Parity

This Mojo implementation covers all core modules from the Python SDK:

| Python Module | Mojo Module | Status |
|--------------|-------------|--------|
| `graph.py` | `graph.mojo` | ✅ Full |
| `laplacian.py` | `laplacian.mojo` | ✅ Full + SIMD |
| `eigen.py` | `eigen.mojo` | ✅ Full (native QR) |
| `conservation.py` | `conservation.mojo` | ✅ Full |
| `tracker.py` | `tracker.mojo` | ✅ Full |
| `anomaly.py` | `conservation.mojo` | ✅ Integrated |
| `fingerprint.py` | `conservation.mojo` | ✅ Integrated |

## Mojo-Specific Features Used

| Feature | Where | Why |
|---------|-------|-----|
| `SIMD[DType.float64, 4]` | `laplacian.mojo` | 4× vectorized matrix ops |
| `UnsafePointer[Float64]` | All modules | Zero-copy, GC-free matrices |
| `fn` (Mojo functions) | Internal paths | Explicit types, no boxing |
| `def` (Python-compatible) | Public API | Ergonomic call interface |
| `owned` parameter | Graph→Laplacian handoff | Move semantics, no copy |
| `borrowed` parameter | Eigen access in analysis | Borrow checking, zero cost |
| `inout` parameter | Graph mutation | Explicit mutability |
| `StringLiteral` parameter | Laplacian type dispatch | Compile-time resolution |
| `DynamicVector` | Edge/vertex storage | Growable, no stdlib dependency |
| `@value` struct decorator | All structs | Auto-generate `__init__`, `__copyinit__` |
| Manual `__del__` | Laplacian, EigenDecomposition | Explicit free of UnsafePointer |

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
