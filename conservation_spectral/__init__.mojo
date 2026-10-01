"""Conservation Spectral SDK — spectral analysis of tension graphs.

Mojo implementation with SIMD-accelerated Laplacian construction,
zero-copy matrix operations, and compile-time graph size specialization.

Ported from conservation-spectral-python.
Verified on-box against Mojo 1.2.0.dev2026100105 (2026-10-01).
"""

from .graph import TensionGraph, Edge
from .laplacian import Laplacian, build_laplacian, build_laplacian_simd
from .eigen import EigenDecomposition, eigendecompose
from .conservation import (
    ConservationRatio, ConservationReport, SpectralFingerprint,
    conservation_ratio, conservation_ratios, spectral_gap,
    cheeger_constant_approx, analyze,
)
from .tracker import ConservationTracker, Alert
