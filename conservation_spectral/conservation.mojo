"""Conservation analysis — ratios, spectral gap, Cheeger constant, full reports.

Mojo implementation with SIMD-accelerated computations and zero-copy
eigenvalue access via UnsafePointer.
"""

from collections.dynamic_vector import DynamicVector
from .graph import TensionGraph
from .laplacian import Laplacian, build_laplacian
from .eigen import EigenDecomposition, eigendecompose


@value
struct ConservationRatio:
    """Conservation ratio for one eigenvector mode."""
    var eigenvector_index: Int
    var eigenvalue: Float64
    var ratio: Float64
    var attribute_name: StringLiteral

    fn __init__(inout self, idx: Int, eval: Float64, ratio: Float64, name: StringLiteral = "default"):
        self.eigenvector_index = idx
        self.eigenvalue = eval
        self.ratio = ratio
        self.attribute_name = name


@value
struct SpectralFingerprint:
    """Summary statistics of the eigenspectrum."""
    var spectral_entropy: Float64
    var effective_dimension: Float64
    var anomaly_count: Int

    fn __init__(inout self):
        self.spectral_entropy = 0.0
        self.effective_dimension = 0.0
        self.anomaly_count = 0


@value
struct ConservationReport:
    """Full conservation analysis report."""
    var ratios: DynamicVector[ConservationRatio]
    var spectral_gap: Float64
    var cheeger_constant: Float64
    var fingerprint: SpectralFingerprint
    var anomaly_count: Int

    fn __init__(inout self):
        self.ratios = DynamicVector[ConservationRatio]()
        self.spectral_gap = 0.0
        self.cheeger_constant = 0.0
        self.fingerprint = SpectralFingerprint()
        self.anomaly_count = 0


fn conservation_ratio(
    eigen: borrowed EigenDecomposition,
    attribute: UnsafePointer[Float64],
    eigenvector_index: Int,
) -> Float64:
    """Compute conservation ratio of an attribute along the k-th eigenvector.

    CR(k) = Var(gradient of attribute projected onto eigenvector_k)
    Low ratio = attribute is well-conserved in this mode.
    """
    let n = eigen.n
    let k = eigenvector_index
    let phi = eigen.eigenvectors + k * n  # k-th eigenvector column

    # Compute projection = phi * attribute (element-wise)
    # Then gradient = diff(projection)
    # Then variance of gradient

    # First pass: compute projection and its gradient in one go
    if n < 2:
        return 1e30  # infinity analog

    var grad_sum: Float64 = 0.0
    var grad_sq_sum: Float64 = 0.0
    let grad_count = n - 1

    for i in range(grad_count):
        # projection[i] = phi[i] * attribute[i]
        # projection[i+1] = phi[i+1] * attribute[i+1]
        # gradient[i] = projection[i+1] - projection[i]
        let p0 = phi.load(i) * attribute.load(i)
        let p1 = phi.load(i + 1) * attribute.load(i + 1)
        let grad = p1 - p0
        grad_sum += grad
        grad_sq_sum += grad * grad

    let mean = grad_sum / Float64(grad_count)
    let variance = grad_sq_sum / Float64(grad_count) - mean * mean
    return variance


fn conservation_ratios(
    eigen: borrowed EigenDecomposition,
    attribute: UnsafePointer[Float64],
    attribute_name: StringLiteral = "default",
) -> DynamicVector[ConservationRatio]:
    """Compute conservation ratios for all eigenvectors."""
    var result = DynamicVector[ConservationRatio]()
    for k in range(eigen.n):
        let r = conservation_ratio(eigen, attribute, k)
        result.push_back(ConservationRatio(
            k,
            eigen.eigenvalues.load(k),
            r,
            attribute_name,
        ))
    return result


fn spectral_gap(eigen: borrowed EigenDecomposition) -> Float64:
    """Compute the spectral gap: largest gap between consecutive eigenvalues.

    Excludes the trivial zero eigenvalue gap.
    """
    let n = eigen.n
    if n < 2:
        return 0.0

    var max_gap: Float64 = 0.0
    for i in range(1, n - 1):  # skip gap 0→1 (trivial eigenvalue)
        let gap = eigen.eigenvalues.load(i + 1) - eigen.eigenvalues.load(i)
        if gap > max_gap:
            max_gap = gap

    return max_gap


fn cheeger_constant_approx(eigen: borrowed EigenDecomposition) -> Float64:
    """Approximate Cheeger constant: h ≈ λ₂ / 2.

    Uses the spectral approximation from the Fiedler vector.
    """
    if eigen.n >= 2:
        return eigen.eigenvalues.load(1) / 2.0
    return 0.0


fn compute_spectral_fingerprint(eigen: borrowed EigenDecomposition) -> SpectralFingerprint:
    """Compute spectral entropy and effective dimension from eigenvalues."""
    var fp = SpectralFingerprint()
    let n = eigen.n

    # Spectral entropy: H = -Σ p_i log(p_i) where p_i = |λ_i| / Σ|λ_i|
    var total: Float64 = 0.0
    for i in range(n):
        total += abs(eigen.eigenvalues.load(i))

    if total < 1e-15:
        return fp

    var entropy: Float64 = 0.0
    for i in range(n):
        let p = abs(eigen.eigenvalues.load(i)) / total
        if p > 1e-15:
            entropy -= p * log(p)

    fp.spectral_entropy = entropy
    fp.effective_dimension = exp(entropy)
    return fp


fn detect_anomalies(
    eigen: borrowed EigenDecomposition,
    graph: borrowed TensionGraph,
    threshold: Float64 = 2.0,
) -> Int:
    """Detect vertices where eigenvector components deviate from the mean.

    Returns the count of anomalous vertices.
    Uses z-score based detection on each eigenvector.
    """
    let n = eigen.n
    if n < 3:
        return 0

    var anomaly_count: Int = 0

    # Check Fiedler vector (index 1) and first few modes
    let modes_to_check = min(3, n)
    for k in range(modes_to_check):
        let phi = eigen.eigenvectors + k * n

        # Compute mean
        var mean: Float64 = 0.0
        for i in range(n):
            mean += phi.load(i)
        mean /= Float64(n)

        # Compute std
        var std: Float64 = 0.0
        for i in range(n):
            let diff = phi.load(i) - mean
            std += diff * diff
        std = sqrt(std / Float64(n))

        if std < 1e-15:
            continue

        # Count outliers
        for i in range(n):
            let z = abs(phi.load(i) - mean) / std
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

    let n = graph.n_vertices
    if n < 2:
        return report

    # Build Laplacian
    let lap = build_laplacian(graph, laplacian_type)

    # Eigendecomposition
    let eigen = eigendecompose(lap, laplacian_type=laplacian_type)

    # Use vertex indices as default attribute
    var default_attr = UnsafePointer[Float64].alloc(n)
    for i in range(n):
        default_attr.store(i, Float64(i))

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

    default_attr.free()

    return report
