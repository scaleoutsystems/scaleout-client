"""Generic priority queue infrastructure for offline-resilient outbound messaging.

Protocol-agnostic: no imports from grpc, proto stubs, or any wire format live here.
Concrete transport adapters live next to their protocol (e.g. scaleout.client.grpc_transport).
"""

from scaleoututil.queue.record import (
    DEFAULT_CLASS_NAME,
    Acked,
    ClassOrder,
    DecayPolicy,
    LinkQuality,
    PermanentFailure,
    PriorityClass,
    QueueRecord,
    QueueStats,
    RecordState,
    SendResult,
    Sent,
    TransientFailure,
)
from scaleoututil.queue.interface import PersistentBackend, TransportAdapter
from scaleoututil.queue.priority_queue import PriorityQueue
from scaleoututil.queue.backends.memory import InMemoryBackend

__all__ = [
    "DEFAULT_CLASS_NAME",
    "Acked",
    "ClassOrder",
    "DecayPolicy",
    "InMemoryBackend",
    "LinkQuality",
    "PermanentFailure",
    "PersistentBackend",
    "PriorityClass",
    "PriorityQueue",
    "QueueRecord",
    "QueueStats",
    "RecordState",
    "SendResult",
    "Sent",
    "TransientFailure",
    "TransportAdapter",
]
