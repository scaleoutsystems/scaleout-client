"""Client-internal link-quality estimator.

Produces HEALTHY | DEGRADED from heartbeat RTT + gRPC channel state.
Never crosses the wire; OFFLINE is CP-only.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import grpc

from scaleoututil.logging import ScaleoutLogger
from scaleoututil.queue.record import LinkQuality


RTT_WINDOW_SIZE = 10
RTT_DEGRADED_MS = 1500.0
RTT_HEALTHY_MS = 500.0
CONFIRM_TO_DEGRADED = 3  # consecutive beats above threshold before flipping
CONFIRM_TO_HEALTHY = 5  # consecutive beats below threshold before restoring
MAX_FAILURE_GAP_S = 30.0  # seconds without a successful beat before forcing DEGRADED
DEFAULT_HEARTBEAT_INTERVAL_S = 2.0


class LinkQualityEstimator:
    """Thread-safe client-side link quality signal.

    Owns the heartbeat loop. Call start(beat_fn) to begin probing; read .quality
    whenever needed. Optionally subscribe to a gRPC channel via attach_channel()
    for proactive TRANSIENT_FAILURE notifications.
    """

    def __init__(self, interval: float = DEFAULT_HEARTBEAT_INTERVAL_S) -> None:
        self._lock = threading.Lock()
        self._quality = LinkQuality.HEALTHY

        self._rtt_window: deque[float] = deque(maxlen=RTT_WINDOW_SIZE)
        self._consecutive_failures = 0
        self._last_success_time = time.monotonic()

        self._bad_streak = 0
        self._good_streak = 0

        self._channel_failed = False
        self._current_channel: Optional[grpc.Channel] = None

        self._interval = interval
        self._beat_fn: Optional[Callable[[], object]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- heartbeat loop --------------------------------------------------------

    def start(self, beat_fn: Callable[[], object]) -> None:
        """Start the heartbeat probe loop.

        beat_fn is invoked once per interval; it should perform a single
        unary RPC and raise on failure. RTT is measured around the call.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._beat_fn = beat_fn
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="link-quality-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the heartbeat loop and unsubscribe from the channel."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self.detach()

    def _heartbeat_loop(self) -> None:
        if self._beat_fn is None:
            raise Exception("_beat_fn is None")
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._beat_fn()
                self.record_heartbeat_rtt((time.monotonic() - t0) * 1000.0)
            except Exception as e:
                self.record_heartbeat_failure()
                ScaleoutLogger().debug(f"Heartbeat failed: {e}")
            self._stop_event.wait(self._interval)

    # -- signal ingestion ------------------------------------------------------

    def record_heartbeat_rtt(self, rtt_ms: float) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._last_success_time = time.monotonic()
            self._channel_failed = False
            self._rtt_window.append(rtt_ms)
            self._recompute()

    def record_heartbeat_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._good_streak = 0
            gap = time.monotonic() - self._last_success_time
            if gap >= MAX_FAILURE_GAP_S:
                self._set_quality(LinkQuality.DEGRADED)

    # -- channel subscription --------------------------------------------------

    def attach_channel(self, channel: grpc.Channel) -> None:
        """Subscribe to channel state changes. Call after channel creation."""
        with self._lock:
            self._current_channel = channel
        channel.subscribe(self._on_channel_state, try_to_connect=False)

    def detach(self) -> None:
        """Unsubscribe. Call on shutdown."""
        with self._lock:
            ch = self._current_channel
            self._current_channel = None
        if ch is not None:
            try:
                ch.unsubscribe(self._on_channel_state)
            except Exception as e:
                ScaleoutLogger().debug(f"LinkQualityEstimator.detach: unsubscribe failed (ignored): {e!r}")

    # -- read current quality --------------------------------------------------

    @property
    def quality(self) -> LinkQuality:
        with self._lock:
            return self._quality

    @property
    def interval_ms(self) -> int:
        return int(self._interval * 1000)

    @property
    def rtt_median_ms(self) -> Optional[float]:
        """Current median RTT in ms, or None if fewer than 2 samples recorded."""
        with self._lock:
            if len(self._rtt_window) < 2:
                return None
            return _median(self._rtt_window)

    # -- gRPC callback (may be called from gRPC internal thread) ---------------

    def _on_channel_state(self, connectivity: grpc.ChannelConnectivity) -> None:
        with self._lock:
            if connectivity == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                self._channel_failed = True
                self._set_quality(LinkQuality.DEGRADED)
            elif connectivity in (grpc.ChannelConnectivity.READY, grpc.ChannelConnectivity.IDLE):
                self._channel_failed = False
                self._recompute()
            # CONNECTING and SHUTDOWN are no-ops

    # -- internal (caller holds self._lock) ------------------------------------

    def _recompute(self) -> None:
        if len(self._rtt_window) < 2:
            return

        median = _median(self._rtt_window)

        if median > RTT_DEGRADED_MS:
            self._bad_streak += 1
            self._good_streak = 0
        elif median < RTT_HEALTHY_MS:
            self._good_streak += 1
            self._bad_streak = 0
        # in the hysteresis band: keep current state, reset neither streak

        if self._bad_streak >= CONFIRM_TO_DEGRADED or self._channel_failed:
            self._set_quality(LinkQuality.DEGRADED)
        elif self._good_streak >= CONFIRM_TO_HEALTHY and not self._channel_failed:
            self._set_quality(LinkQuality.HEALTHY)

    def _set_quality(self, new_quality: LinkQuality) -> None:
        if new_quality == self._quality:
            return
        if new_quality == LinkQuality.DEGRADED:
            self._good_streak = 0
        self._quality = new_quality
        ScaleoutLogger().info(f"LinkQualityEstimator: quality changed to {new_quality.value}")


def _median(window: deque) -> float:
    sorted_vals = sorted(window)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return float(sorted_vals[mid])
