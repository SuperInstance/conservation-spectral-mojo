"""Conservation analysis — ratios, spectral gap, Cheeger constant, full reports.

Mojo port (verified on-box 2026-10-01): List (ex-DynamicVector),
Pointer[Float64, MutUntrackedOrigin] (ex-UnsafePointer), def-only.
"""

from std import time
from std.math import log, exp, sqrt

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import Laplacian, build_laplacian
from conservation_spectral.eigen import EigenDecomposition, eigendecompose


struct ConservationRatio:
    """Conservation ratio for one eigenvector mode."""
    var eigenvector_index: Int
    var eigenvalue: Float64
    var ratio: Float64
    var attribute_name: String

    def __init__(out self, idx: Int, eval_: Float64, ratio: Float64, name: String = "default"):
        self.eigenvector_index = idx
        self.eigenvalue = eval_
        self.ratio = ratio
        self.attribute_name = name


struct SpectralFingerprint:
    """Summary statistics of the eigenspectrum."""
    var spectral_entropy: Float64
    var effective_dimension: Float64
    var anomaly_count: Int

    def __init__(out self):
        self.spectral_entropy = 0.0
        self.effective_dimension = 0.0
        self.anomaly_count = 0


struct ConservationReport:
    """Full conservation analysis report."""
    var ratios: List[ConservationRatio]
    var spectral_gap: Float64
    var cheeger_constant: Float64
    var fingerprint: SpectralFingerprint
    var anomaly_count: Int

    def __init__(out self):
        self.ratios = List[ConservationRatio]()
        self.spectral_gap = 0.0
        self.cheeger_constant = 0.0
        self.fingerprint = SpectralFingerprint()
        self.anomaly_count = 0


def conservation_ratio(
    eigen: EigenDecomposition,
    attribute: Pointer[Float64, MutUntrackedOrigin],
    eigenvector_index: Int,
) -> Float64:
    """Compute conservation ratio of an attribute along the k-th eigenvector.

    CR(k) = Var(gradient of attribute projected onto eigenvector_k)
    Low ratio = attribute is well-conserved in this mode.
    """
    var n = eigen.n
    var k = eigenvector_index
    var phi = eigen.eigenvectors.unsafe_offset(k * n)  # k-th eigenvector column

    # Compute projection = phi * attribute (element-wise)
    # Then gradient = diff(projection)
    # Then variance of gradient

    # First pass: compute projection and its gradient in one go
    if n < 2:
        return 1e30  # infinity analog

    var grad_sum: Float64 = 0.0
    var grad_sq_sum: Float64 = 0.0
    var grad_count = n - 1

    for i in range(grad_count):
        # projection[i] = phi[i] * attribute[i]
        # projection[i+1] = phi[i+1] * attribute[i+1]
        # gradient[i] = projection[i+1] - projection[i]
        var p0 = phi.unsafe_load(i) * attribute.unsafe_load(i)
        var p1 = phi.unsafe_load(i + 1) * attribute.unsafe_load(i + 1)
        var grad = p1 - p0
        grad_sum += grad
        grad_sq_sum += grad * grad

    var mean = grad_sum / Float64(grad_count)
    var variance = grad_sq_sum / Float64(grad_count) - mean * mean
    return variance


def conservation_ratios(
    eigen: EigenDecomposition,
    attribute: Pointer[Float64, MutUntrackedOrigin],
    attribute_name: StringLiteral = "default",
) -> List[ConservationRatio]:
    """Compute conservation ratios for all eigenvectors."""
    var result = List[ConservationRatio]()
    for k in range(eigen.n):
        var r = conservation_ratio(eigen, attribute, k)
        result.append(ConservationRatio(
            k,
            eigen.eigenvalues.unsafe_load(k),
            r,
            attribute_name,
        ))
    return result^


def spectral_gap(eigen: EigenDecomposition) -> Float64:
    """Compute the spectral gap: largest gap between consecutive eigenvalues.

    Excludes the trivial zero eigenvalue gap.
    """
    var n = eigen.n
    if n < 2:
        return 0.0

    var max_gap: Float64 = 0.0
    for i in range(1, n - 1):  # skip gap 0→1 (trivial eigenvalue)
        var gap = eigen.eigenvalues.unsafe_load(i + 1) - eigen.eigenvalues.unsafe_load(i)
        if gap > max_gap:
            max_gap = gap

    return max_gap


def cheeger_constant_approx(eigen: EigenDecomposition) -> Float64:
    """Approximate Cheeger constant: h ≈ λ₂ / 2.

    Uses the spectral approximation from the Fiedler vector.
    """
    if eigen.n >= 2:
        return eigen.eigenvalues.unsafe_load(1) / 2.0
    return 0.0


def compute_spectral_fingerprint(eigen: EigenDecomposition) -> SpectralFingerprint:
    """Compute spectral entropy and effective dimension from eigenvalues."""
    var fp = SpectralFingerprint()
    var n = eigen.n

    # Spectral entropy: H = -Σ p_i log(p_i) where p_i = |λ_i| / Σ|λ_i|
    var total: Float64 = 0.0
    for i in range(n):
        total += abs(eigen.eigenvalues.unsafe_load(i))

    if total < 1e-15:
        return fp^

    var entropy: Float64 = 0.0
    for i in range(n):
        var p = abs(eigen.eigenvalues.unsafe_load(i)) / total
        if p > 1e-15:
            entropy -= p * log(p)

    fp.spectral_entropy = entropy
    fp.effective_dimension = exp(entropy)
    return fp^


def detect_anomalies(
    eigen: EigenDecomposition,
    graph: TensionGraph,
    threshold: Float64 = 2.0,
) -> Int:
    """Detect vertices where eigenvector components deviate from the mean.

    Returns the count of anomalous vertices.
    Uses z-score based detection on each eigenvector.
    """
    var n = eigen.n
    if n < 3:
        return 0

    var anomaly_count: Int = 0

    # Check Fiedler vector (index 1) and first few modes
    var modes_to_check = min(3, n)
    for k in range(modes_to_check):
        var phi = eigen.eigenvectors.unsafe_offset(k * n)

        # Compute mean
        var mean: Float64 = 0.0
        for i in range(n):
            mean += phi.unsafe_load(i)
        mean /= Float64(n)

        # Compute std
        var std: Float64 = 0.0
        for i in range(n):
            var diff = phi.unsafe_load(i) - mean
            std += diff * diff
        std = sqrt(std / Float64(n))

        if std < 1e-15:
            continue

        # Count outliers
        for i in range(n):
            var z = abs(phi.unsafe_load(i) - mean) / std
            if z > threshold:
                anomaly_count += 1

    return anomaly_count


def analyze(
    graph: TensionGraph,
    laplacian_type: StringLiteral = "symmetric_normalized",
) -> ConservationReport:
    """One-call convenience: graph → Laplacian → eigendecomposition → report.

    Args:
        graph: TensionGraph to analyze.
        laplacian_type: Type of Laplacian to build.

    Returns:
        ConservationReport with full analysis.
    """
    var report = ConservationReport()

    var n = graph.n_vertices
    if n < 2:
        return report^

    # Build Laplacian
    var lap = build_laplacian(graph, laplacian_type)

    # Eigendecomposition
    var eigen = eigendecompose(lap, 0, laplacian_type)

    # Use vertex indices as default attribute
    var default_attr = alloc[Float64](n)
    for i in range(n):
        default_attr.unsafe_store(i, Float64(i))

    # Conservation ratios
    report.ratios = conservation_ratios(eigen, default_attr, "vertex_index")

    # Spectral gap
    report.spectral_gap = spectral_gap(eigen)

    # Cheeger constant
    report.cheeger_constant = cheeger_constant_approx(eigen)

    # Spectral fingerprint
    report.fingerprint = compute_spectral_fingerprint(eigen)

    # Anomaly detection
    report.anomaly_count = detect_anomalies(eigen, graph, 2.0)
    report.fingerprint.anomaly_count = report.anomaly_count

    default_attr.unsafe_free()

    return report^
