"""Envelope and value types for the generic priority queue."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class RecordState(str, Enum):
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    SENT = "sent"
    ACKED = "acked"
    FAILED = "failed"


class LinkQuality(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class ClassOrder(str, Enum):
    FIFO = "fifo"
    LIFO = "lifo"


# The well-known name of the default catch-all class. It is always present in a backend,
# cannot be removed, and receives records whose declared class isn't registered (typos,
# class removed mid-flight, demote target missing, etc.). Users can override its settings
# at backend construction by passing their own PriorityClass with this name.
DEFAULT_CLASS_NAME = "__default__"


@dataclass(frozen=True)
class DecayPolicy:
    """Age-based policies applied to records of a priority class.

    drop_after: discard records older than this many seconds (wall-clock).
        Evaluated on read against the oldest end of the class deque.

    demote_after: (age_seconds, target_class_name) — once a record's age exceeds
        age_seconds, it is reassigned (moved) to the target class and inherits the
        target's policies. If the target class doesn't exist, the record goes to
        the default class. Most useful for FIFO classes (where the head is the
        oldest record); for LIFO classes the head is the newest, so demotion only
        triggers when even the freshest record is stale.
    """

    drop_after: Optional[float] = None
    demote_after: Optional[tuple[float, str]] = None


@dataclass(frozen=True)
class PriorityClass:
    """One priority class registered in the backend.

    name:      identifier referenced by QueueRecord.priority (e.g. "P5", "telemetry_bulk").
    priority:  scheduling weight. Higher = more urgent. Recommended range 0..100,
               not enforced — leave room to interleave new classes later.
    order:     FIFO or LIFO drain order within the class.
    capacity:  if set, drop-oldest within this class once depth reaches the cap.
    decay:     optional DecayPolicy for age-based drop/demote behaviour.
    """

    name: str
    priority: int
    order: ClassOrder
    capacity: Optional[int] = None
    decay: Optional[DecayPolicy] = None


@dataclass(frozen=True)
class QueueRecord:
    record_id: str
    priority: str
    message_type: str
    payload: bytes
    enqueued_at: float
    attempts: int = 0
    state: RecordState = RecordState.QUEUED

    @classmethod
    def create(cls, message_type: str, payload: bytes, priority: str) -> "QueueRecord":
        # Wall-clock so enqueued_at is meaningful across process restarts and across
        # peer comparison — required by DecayPolicy and by a future SQLite backend.
        # FIFO/LIFO ordering doesn't depend on this value (it uses deque insertion order),
        # so NTP step adjustments only affect decay decisions, not delivery order.
        return cls(
            record_id=str(uuid.uuid4()),
            priority=priority,
            message_type=message_type,
            payload=payload,
            enqueued_at=time.time(),
        )


@dataclass(frozen=True)
class Acked:
    """Transport completed synchronously; server confirmed receipt inline.

    Drainer marks the record ACKED (terminal). Used by unary/request-response
    adapters where the response itself is the ack.
    """

    pass


@dataclass(frozen=True)
class Sent:
    """Adapter accepted the record; ack is deferred (streaming transports).

    Drainer marks the record SENT; the adapter calls backend.mark_acked()
    asynchronously when the server confirms.
    """

    pass


@dataclass(frozen=True)
class TransientFailure:
    reason: str


@dataclass(frozen=True)
class PermanentFailure:
    reason: str


SendResult = Union[Acked, Sent, TransientFailure, PermanentFailure]


@dataclass(frozen=True)
class QueueStats:
    depth: int
    in_flight: int
    dropped: int
