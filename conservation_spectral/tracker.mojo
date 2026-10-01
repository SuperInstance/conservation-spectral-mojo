"""Real-time conservation tracking with sliding window.

Mojo port (verified on-box 2026-10-01): Pointer[Float64, MutUntrackedOrigin]
(ex-UnsafePointer — null slots become alloc(1) dummies gated by _has_baseline),
List (ex-DynamicVector), String (ex-StringRef).
"""

from std import time
from std.math import sqrt

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian, build_laplacian_simd
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import (
    conservation_ratios, spectral_gap, cheeger_constant_approx,
    ConservationReport, SpectralFingerprint, compute_spectral_fingerprint,
    detect_anomalies, analyze,
)


struct Alert:
    """Alert from conservation drop detection."""
    var timestamp: Int        # observation index
    var deviation: Float64    # max sigma deviation
    var message: String
    var n_ratios: Int         # number of ratios at alert time

    def __init__(out self, ts: Int, dev: Float64, msg: String, n: Int = 0):
        self.timestamp = ts
        self.deviation = dev
        self.message = msg
        self.n_ratios = n


struct ConservationTracker:
    """Sliding-window tracker for real-time conservation monitoring.

    Maintains a window of recent observations and flags when conservation
    ratios deviate significantly from the baseline.
    """
    var window_size: Int
    var observation_count: Int
    var _observations: List[Pointer[Float64, MutUntrackedOrigin]]
    var _obs_lengths: List[Int]
    var _baseline_ratios: Pointer[Float64, MutUntrackedOrigin]
    var _baseline_std: Pointer[Float64, MutUntrackedOrigin]
    var _baseline_len: Int
    var _current_ratios: Pointer[Float64, MutUntrackedOrigin]
    var _current_len: Int
    var _ratio_history: List[Pointer[Float64, MutUntrackedOrigin]]
    var _ratio_history_lens: List[Int]
    var _has_baseline: Bool

    def __init__(out self, window_size: Int = 100):
        self.window_size = window_size
        self.observation_count = 0
        self._observations = List[Pointer[Float64, MutUntrackedOrigin]]()
        self._obs_lengths = List[Int]()
        self._baseline_ratios = alloc[Float64](1)  # dummy until established
        self._baseline_std = alloc[Float64](1)     # dummy until established
        self._baseline_len = 0
        self._current_ratios = alloc[Float64](1)   # dummy until fed
        self._current_len = 0
        self._ratio_history = List[Pointer[Float64, MutUntrackedOrigin]]()
        self._ratio_history_lens = List[Int]()
        self._has_baseline = False

    def feed(mut self, observation: Pointer[Float64, MutUntrackedOrigin], obs_len: Int) -> Optional[Alert]:
        """Feed a new observation. Returns an Alert if conservation drops.

        Args:
            observation: pointer to 1-D array of values.
            obs_len: length of the observation array.

        Returns:
            Alert if anomaly detected, None otherwise.
        """
        # Copy observation data
        var obs_copy = alloc[Float64](obs_len)
        for i in range(obs_len):
            obs_copy.unsafe_store(i, observation.unsafe_load(i))

        self._observations.append(obs_copy)
        self._obs_lengths.append(obs_len)
        self.observation_count += 1

        # Keep sliding window
        if len(self._observations) > self.window_size:
            var old = self._observations[0]
            old.unsafe_free()
            # Shift everything left
            var new_obs = List[Pointer[Float64, MutUntrackedOrigin]]()
            var new_lens = List[Int]()
            for i in range(1, len(self._observations)):
                new_obs.append(self._observations[i])
                new_lens.append(self._obs_lengths[i])
            self._observations = new_obs^
            self._obs_lengths = new_lens^

        # Need at least 3 observations
        if len(self._observations) < 3:
            return None

        # Build transition graph from window
        var transitions = List[Tuple[Int, Int]]()
        for i in range(len(self._observations) - 1):
            # Find argmax of consecutive observations
            var src_max_idx: Int = 0
            var src_max_val: Float64 = self._observations[i].unsafe_load(0)
            for j in range(1, self._obs_lengths[i]):
                if self._observations[i].unsafe_load(j) > src_max_val:
                    src_max_val = self._observations[i].unsafe_load(j)
                    src_max_idx = j

            var tgt_max_idx: Int = 0
            var tgt_max_val: Float64 = self._observations[i + 1].unsafe_load(0)
            for j in range(1, self._obs_lengths[i + 1]):
                if self._observations[i + 1].unsafe_load(j) > tgt_max_val:
                    tgt_max_val = self._observations[i + 1].unsafe_load(j)
                    tgt_max_idx = j

            transitions.append(Tuple(src_max_idx, tgt_max_idx))

        if len(transitions) == 0:
            return None

        var graph = TensionGraph.build_from_transitions(transitions)
        if graph.n_vertices < 2:
            return None

        # Compute conservation
        var lap = build_laplacian(graph)
        var eigen = eigendecompose(lap)

        # Uniform attribute
        var attr = alloc[Float64](graph.n_vertices)
        for i in range(graph.n_vertices):
            attr.unsafe_store(i, 1.0)

        var ratios = conservation_ratios(eigen, attr, "tracking")
        attr.unsafe_free()

        # Store current ratios (note: previous _current_ratios buffer is
        # intentionally not freed here — original behavior preserved)
        self._current_len = len(ratios)
        self._current_ratios = alloc[Float64](len(ratios))
        for i in range(len(ratios)):
            self._current_ratios.unsafe_store(i, ratios[i].ratio)

        # Save to history
        var hist_copy = alloc[Float64](len(ratios))
        for i in range(len(ratios)):
            hist_copy.unsafe_store(i, ratios[i].ratio)
        self._ratio_history.append(hist_copy)
        self._ratio_history_lens.append(len(ratios))

        # Establish baseline after enough observations
        if not self._has_baseline and len(self._observations) >= min(10, self.window_size // 2):
            self._establish_baseline()

        # Check for alert
        if self._has_baseline:
            return self._check_alert()

        return None

    def _establish_baseline(mut self):
        """Set baseline from accumulated ratio history."""
        if len(self._ratio_history) < 3:
            return

        var max_len = self._max_ratio_len()
        if max_len == 0:
            return

        self._baseline_len = max_len
        self._baseline_ratios = alloc[Float64](max_len)
        self._baseline_std = alloc[Float64](max_len)

        # Compute mean
        for j in range(max_len):
            var sum: Float64 = 0.0
            var count: Int = 0
            for i in range(len(self._ratio_history)):
                if j < self._ratio_history_lens[i]:
                    sum += self._ratio_history[i].unsafe_load(j)
                    count += 1
            self._baseline_ratios.unsafe_store(j, sum / Float64(count) if count > 0 else 0.0)

        # Compute std
        for j in range(max_len):
            var sq_sum: Float64 = 0.0
            var count: Int = 0
            var mean = self._baseline_ratios.unsafe_load(j)
            for i in range(len(self._ratio_history)):
                if j < self._ratio_history_lens[i]:
                    var diff = self._ratio_history[i].unsafe_load(j) - mean
                    sq_sum += diff * diff
                    count += 1
            self._baseline_std.unsafe_store(j, sqrt(sq_sum / Float64(count)) if count > 1 else 1.0)

        self._has_baseline = True

    def _max_ratio_len(self) -> Int:
        var max_len: Int = 0
        for i in range(len(self._ratio_history_lens)):
            if self._ratio_history_lens[i] > max_len:
                max_len = self._ratio_history_lens[i]
        return max_len

    def _check_alert(mut self) -> Optional[Alert]:
        """Check if current ratios deviate from baseline."""
        var max_len = max(self._current_len, self._baseline_len)
        var max_dev: Float64 = 0.0
        var threshold: Float64 = 2.0

        for j in range(max_len):
            var cur = self._current_ratios.unsafe_load(j) if j < self._current_len else 0.0
            var base = self._baseline_ratios.unsafe_load(j) if j < self._baseline_len else 0.0
            var std = self._baseline_std.unsafe_load(j) if j < self._baseline_len else 1.0
            if std > 1e-12:
                var dev = abs(cur - base) / std
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

    def report(mut self) -> Optional[ConservationReport]:
        """Generate a full ConservationReport from the current window."""
        if len(self._observations) < 3:
            return None

        var transitions = List[Tuple[Int, Int]]()
        for i in range(len(self._observations) - 1):
            var src_max_idx: Int = 0
            var src_max_val: Float64 = self._observations[i].unsafe_load(0)
            for j in range(1, self._obs_lengths[i]):
                if self._observations[i].unsafe_load(j) > src_max_val:
                    src_max_val = self._observations[i].unsafe_load(j)
                    src_max_idx = j

            var tgt_max_idx: Int = 0
            var tgt_max_val: Float64 = self._observations[i + 1].unsafe_load(0)
            for j in range(1, self._obs_lengths[i + 1]):
                if self._observations[i + 1].unsafe_load(j) > tgt_max_val:
                    tgt_max_val = self._observations[i + 1].unsafe_load(j)
                    tgt_max_idx = j

            transitions.append(Tuple(src_max_idx, tgt_max_idx))

        if len(transitions) == 0:
            return None

        var graph = TensionGraph.build_from_transitions(transitions)
        if graph.n_vertices < 2:
            return None

        return analyze(graph)

    def reset(mut self):
        """Reset tracker state."""
        # Free observation memory
        for i in range(len(self._observations)):
            self._observations[i].unsafe_free()
        self._observations = List[Pointer[Float64, MutUntrackedOrigin]]()
        self._obs_lengths = List[Int]()

        if self._has_baseline:
            self._baseline_ratios.unsafe_free()
            self._baseline_std.unsafe_free()
        self._baseline_len = 0
        self._has_baseline = False

        if self._current_len > 0:
            self._current_ratios.unsafe_free()
        self._current_len = 0

        for i in range(len(self._ratio_history)):
            self._ratio_history[i].unsafe_free()
        self._ratio_history = List[Pointer[Float64, MutUntrackedOrigin]]()
        self._ratio_history_lens = List[Int]()

        self.observation_count = 0
