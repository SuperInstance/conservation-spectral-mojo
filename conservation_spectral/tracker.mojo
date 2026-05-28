"""Real-time conservation tracking with sliding window.

Mojo implementation with efficient memory management via UnsafePointer
and DynamicVector for the observation buffer.
"""

from collections.dynamic_vector import DynamicVector
from .graph import TensionGraph
from .laplacian import build_laplacian, build_laplacian_simd
from .eigen import eigendecompose
from .conservation import (
    conservation_ratios, spectral_gap, cheeger_constant_approx,
    ConservationReport, SpectralFingerprint, compute_spectral_fingerprint,
    detect_anomalies, analyze,
)


@value
struct Alert:
    """Alert from conservation drop detection."""
    var timestamp: Int        # observation index
    var deviation: Float64    # max sigma deviation
    var message: StringRef
    var n_ratios: Int         # number of ratios at alert time

    fn __init__(inout self, ts: Int, dev: Float64, msg: StringRef, n: Int = 0):
        self.timestamp = ts
        self.deviation = dev
        self.message = msg
        self.n_ratios = n


@value
struct ConservationTracker:
    """Sliding-window tracker for real-time conservation monitoring.

    Maintains a window of recent observations and flags when conservation
    ratios deviate significantly from the baseline.
    """
    var window_size: Int
    var observation_count: Int
    var _observations: DynamicVector[UnsafePointer[Float64]]
    var _obs_lengths: DynamicVector[Int]
    var _baseline_ratios: UnsafePointer[Float64]
    var _baseline_std: UnsafePointer[Float64]
    var _baseline_len: Int
    var _current_ratios: UnsafePointer[Float64]
    var _current_len: Int
    var _ratio_history: DynamicVector[UnsafePointer[Float64]]
    var _ratio_history_lens: DynamicVector[Int]
    var _has_baseline: Bool

    fn __init__(inout self, window_size: Int = 100):
        self.window_size = window_size
        self.observation_count = 0
        self._observations = DynamicVector[UnsafePointer[Float64]]()
        self._obs_lengths = DynamicVector[Int]()
        self._baseline_ratios = UnsafePointer[Float64](0)
        self._baseline_std = UnsafePointer[Float64](0)
        self._baseline_len = 0
        self._current_ratios = UnsafePointer[Float64](0)
        self._current_len = 0
        self._ratio_history = DynamicVector[UnsafePointer[Float64]]()
        self._ratio_history_lens = DynamicVector[Int]()
        self._has_baseline = False

    fn feed(inout self, observation: UnsafePointer[Float64], obs_len: Int) -> Optional[Alert]:
        """Feed a new observation. Returns an Alert if conservation drops.

        Args:
            observation: pointer to 1-D array of values.
            obs_len: length of the observation array.

        Returns:
            Alert if anomaly detected, None otherwise.
        """
        # Copy observation data
        var obs_copy = UnsafePointer[Float64].alloc(obs_len)
        for i in range(obs_len):
            obs_copy.store(i, observation.load(i))

        self._observations.push_back(obs_copy)
        self._obs_lengths.push_back(obs_len)
        self.observation_count += 1

        # Keep sliding window
        if self._observations.size > self.window_size:
            let old = self._observations[0]
            old.free()
            # Shift everything left
            var new_obs = DynamicVector[UnsafePointer[Float64]]()
            var new_lens = DynamicVector[Int]()
            for i in range(1, self._observations.size):
                new_obs.push_back(self._observations[i])
                new_lens.push_back(self._obs_lengths[i])
            self._observations = new_obs
            self._obs_lengths = new_lens

        # Need at least 3 observations
        if self._observations.size < 3:
            return None

        # Build transition graph from window
        let n = self._obs_lengths[0]
        var transitions = DynamicVector[Tuple[Int, Int]]()
        for i in range(self._observations.size - 1):
            # Find argmax of consecutive observations
            var src_max_idx: Int = 0
            var src_max_val: Float64 = self._observations[i].load(0)
            for j in range(1, self._obs_lengths[i]):
                if self._observations[i].load(j) > src_max_val:
                    src_max_val = self._observations[i].load(j)
                    src_max_idx = j

            var tgt_max_idx: Int = 0
            var tgt_max_val: Float64 = self._observations[i + 1].load(0)
            for j in range(1, self._obs_lengths[i + 1]):
                if self._observations[i + 1].load(j) > tgt_max_val:
                    tgt_max_val = self._observations[i + 1].load(j)
                    tgt_max_idx = j

            transitions.push_back(Tuple(src_max_idx, tgt_max_idx))

        if transitions.size == 0:
            return None

        let graph = TensionGraph.build_from_transitions(transitions)
        if graph.n_vertices < 2:
            return None

        # Compute conservation
        let lap = build_laplacian(graph)
        let eigen = eigendecompose(lap)

        # Uniform attribute
        var attr = UnsafePointer[Float64].alloc(graph.n_vertices)
        for i in range(graph.n_vertices):
            attr.store(i, 1.0)

        let ratios = conservation_ratios(eigen, attr, "tracking")
        attr.free()

        # Store current ratios
        self._current_len = ratios.size
        self._current_ratios = UnsafePointer[Float64].alloc(ratios.size)
        for i in range(ratios.size):
            self._current_ratios.store(i, ratios[i].ratio)

        # Save to history
        var hist_copy = UnsafePointer[Float64].alloc(ratios.size)
        for i in range(ratios.size):
            hist_copy.store(i, ratios[i].ratio)
        self._ratio_history.push_back(hist_copy)
        self._ratio_history_lens.push_back(ratios.size)

        # Establish baseline after enough observations
        if not self._has_baseline and self._observations.size >= min(10, self.window_size // 2):
            self._establish_baseline()

        # Check for alert
        if self._has_baseline:
            return self._check_alert()

        return None

    fn _establish_baseline(inout self):
        """Set baseline from accumulated ratio history."""
        if self._ratio_history.size < 3:
            return

        let max_len = self._max_ratio_len()
        if max_len == 0:
            return

        self._baseline_len = max_len
        self._baseline_ratios = UnsafePointer[Float64].alloc(max_len)
        self._baseline_std = UnsafePointer[Float64].alloc(max_len)

        # Compute mean
        for j in range(max_len):
            var sum: Float64 = 0.0
            var count: Int = 0
            for i in range(self._ratio_history.size):
                if j < self._ratio_history_lens[i]:
                    sum += self._ratio_history[i].load(j)
                    count += 1
            self._baseline_ratios.store(j, sum / Float64(count) if count > 0 else 0.0)

        # Compute std
        for j in range(max_len):
            var sq_sum: Float64 = 0.0
            var count: Int = 0
            let mean = self._baseline_ratios.load(j)
            for i in range(self._ratio_history.size):
                if j < self._ratio_history_lens[i]:
                    let diff = self._ratio_history[i].load(j) - mean
                    sq_sum += diff * diff
                    count += 1
            self._baseline_std.store(j, sqrt(sq_sum / Float64(count)) if count > 1 else 1.0)

        self._has_baseline = True

    fn _max_ratio_len(self) -> Int:
        var max_len: Int = 0
        for i in range(self._ratio_history_lens.size):
            if self._ratio_history_lens[i] > max_len:
                max_len = self._ratio_history_lens[i]
        return max_len

    fn _check_alert(inout self) -> Optional[Alert]:
        """Check if current ratios deviate from baseline."""
        let max_len = max(self._current_len, self._baseline_len)
        var max_dev: Float64 = 0.0
        let threshold: Float64 = 2.0

        for j in range(max_len):
            let cur = self._current_ratios.load(j) if j < self._current_len else 0.0
            let base = self._baseline_ratios.load(j) if j < self._baseline_len else 0.0
            let std = self._baseline_std.load(j) if j < self._baseline_len else 1.0
            if std > 1e-12:
                let dev = abs(cur - base) / std
                if dev > max_dev:
                    max_dev = dev

        if max_dev > threshold:
            return Alert(
                self.observation_count,
                max_dev,
                "Conservation drop detected",
                self._current_len,
            )
        return None

    def report(inout self) -> Optional[ConservationReport]:
        """Generate a full ConservationReport from the current window."""
        if self._observations.size < 3:
            return None

        let n = self._obs_lengths[0]
        var transitions = DynamicVector[Tuple[Int, Int]]()
        for i in range(self._observations.size - 1):
            var src_max_idx: Int = 0
            var src_max_val: Float64 = self._observations[i].load(0)
            for j in range(1, self._obs_lengths[i]):
                if self._observations[i].load(j) > src_max_val:
                    src_max_val = self._observations[i].load(j)
                    src_max_idx = j

            var tgt_max_idx: Int = 0
            var tgt_max_val: Float64 = self._observations[i + 1].load(0)
            for j in range(1, self._obs_lengths[i + 1]):
                if self._observations[i + 1].load(j) > tgt_max_val:
                    tgt_max_val = self._observations[i + 1].load(j)
                    tgt_max_idx = j

            transitions.push_back(Tuple(src_max_idx, tgt_max_idx))

        if transitions.size == 0:
            return None

        var graph = TensionGraph.build_from_transitions(transitions)
        if graph.n_vertices < 2:
            return None

        return analyze(graph)

    def reset(inout self):
        """Reset tracker state."""
        # Free observation memory
        for i in range(self._observations.size):
            self._observations[i].free()
        self._observations = DynamicVector[UnsafePointer[Float64]]()
        self._obs_lengths = DynamicVector[Int]()

        if self._has_baseline:
            self._baseline_ratios.free()
            self._baseline_std.free()
        self._baseline_len = 0
        self._has_baseline = False

        if self._current_len > 0 and self._current_ratios:
            self._current_ratios.free()
        self._current_len = 0

        for i in range(self._ratio_history.size):
            self._ratio_history[i].free()
        self._ratio_history = DynamicVector[UnsafePointer[Float64]]()
        self._ratio_history_lens = DynamicVector[Int]()

        self.observation_count = 0
