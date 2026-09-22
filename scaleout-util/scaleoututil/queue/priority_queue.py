"""Single-drainer priority queue.

Producers call enqueue() (non-blocking). One drainer thread pulls from the backend
in strict priority order and hands records to the TransportAdapter.
"""

from __future__ import annotations

import threading
from typing import Optional

from scaleoututil.queue.interface import PersistentBackend, TransportAdapter
from scaleoututil.queue.record import (
    Acked,
    PermanentFailure,
    PriorityClass,
    QueueRecord,
    Sent,
    TransientFailure,
)
from scaleoututil.logging import ScaleoutLogger


_MAX_BACKOFF_S = 60.0
_INITIAL_BACKOFF_S = 0.5


class PriorityQueue:
    """Strict-priority queue with a single drainer thread.

    The queue is protocol-agnostic. The transport adapter encapsulates protocol-specific
    encoding, dispatch, and result translation.

    Construction:
        queue = PriorityQueue(
            transport=GrpcTransportAdapter(handler),
            backend=InMemoryBackend(
                classes=[
                    PriorityClass(name="P1", priority=100, order=FIFO),
                    PriorityClass(name="P5", priority=40,  order=LIFO, capacity=10_000),
                ],
            ),
        )
        queue.start()
        ...
        queue.enqueue(record)   # non-blocking
        ...
        queue.stop()
    """

    def __init__(
        self,
        transport: TransportAdapter,
        backend: PersistentBackend,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0

    def enqueue(self, record: QueueRecord) -> str:
        """Append record to its priority class. Non-blocking. Returns record_id."""
        self._backend.append(record)
        self._wake.set()
        return record.record_id

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # restart-safety: anything left as IN_FLIGHT from a prior process goes back to QUEUED
        self._backend.requeue_inflight()
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, name="queue-drainer", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        self._wake.set()
        self._reconnect.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def wake_drainer(self) -> None:
        """Signal that the channel was reconnected — interrupt backoff so the drainer retries sooner."""
        self._reconnect.set()

    # ---- dynamic class registry (pass-through to backend) --------------------

    def add_class(self, cls: PriorityClass) -> None:
        self._backend.add_class(cls)

    def remove_class(self, name: str, force: bool = False) -> int:
        return self._backend.remove_class(name, force=force)

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            swept = self._backend.sweep_unacked()
            if swept:
                ScaleoutLogger().debug(f"queue: requeued {len(swept)} unacked records past TTL")
                self._transport.on_requeue(swept)
            record = self._backend.next_record()
            if record is None:
                # Not infinte wait to avoid race cond that could occur
                self._wake.clear()
                self._wake.wait(5)
                continue

            result = self._send_with_guard(record)

            if isinstance(result, Acked):
                # unary semantics: response received → terminal
                self._backend.mark_acked(record.record_id)
                self._consecutive_failures = 0
            elif isinstance(result, Sent):
                # streaming semantics: on the wire, ack arrives later via mark_acked()
                self._backend.mark_sent(record.record_id)
                self._consecutive_failures = 0
            elif isinstance(result, TransientFailure):
                self._backend.mark_failed(record.record_id, retry=True)
                self._consecutive_failures += 1
                self._backoff_sleep()
            elif isinstance(result, PermanentFailure):
                self._backend.mark_failed(record.record_id, retry=False)
                self._consecutive_failures = 0
            else:
                # adapter returned something unexpected — treat as permanent
                self._backend.mark_failed(record.record_id, retry=False)
                self._consecutive_failures = 0

    def _send_with_guard(self, record: QueueRecord):
        try:
            return self._transport.send(record)
        except Exception as e:
            return TransientFailure(f"adapter raised: {e!r}")

    def _backoff_sleep(self) -> None:
        delay = min(_MAX_BACKOFF_S, _INITIAL_BACKOFF_S * (2 ** (self._consecutive_failures - 1)))
        # wake early on stop() or on a reconnect signal from the transport
        self._reconnect.clear()
        self._reconnect.wait(timeout=delay)
