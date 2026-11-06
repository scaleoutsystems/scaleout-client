"""Contracts the PriorityQueue depends on.

Both are Protocols (structural typing): any class with the right shape satisfies them,
no inheritance required.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from scaleoututil.queue.record import PriorityClass, QueueRecord, QueueStats, SendResult


@runtime_checkable
class TransportAdapter(Protocol):
    """Sends a single record over a concrete protocol.

    Implementations live next to their protocol (e.g. scaleoututil/grpc/grpc_transport.py).
    The adapter owns encoding/decoding, endpoint dispatch, and result translation.
    """

    def send(self, record: QueueRecord) -> SendResult: ...

    def on_requeue(self, record_ids: list[str]) -> None: ...


@runtime_checkable
class PersistentBackend(Protocol):
    """Stores records and decides scheduling order (strict priority + per-class FIFO/LIFO).

    Class registration is dynamic — add/remove at runtime — and the backend always
    maintains an undeletable default class to absorb orphaned records.

    Backpressure (drop-oldest within class) and age-based decay (drop/demote) live
    here too, since both depend on backend-specific state tracking.
    """

    def append(self, record: QueueRecord) -> None: ...

    def next_record(self) -> Optional[QueueRecord]: ...

    def mark_sent(self, record_id: str) -> None: ...

    def mark_acked(self, record_id: str) -> None: ...

    def mark_failed(self, record_id: str, retry: bool) -> None: ...

    def requeue_inflight(self) -> None: ...

    def sweep_unacked(self) -> list[str]: ...

    def stats(self) -> dict[str, QueueStats]: ...

    def add_class(self, cls: PriorityClass) -> None: ...

    def remove_class(self, name: str, force: bool = False) -> int: ...

    def replace_class(self, cls: PriorityClass) -> None: ...
