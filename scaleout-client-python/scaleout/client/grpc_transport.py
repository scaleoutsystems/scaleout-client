"""gRPC TransportAdapter for the generic priority queue.

Wraps an existing GrpcHandler (constructor injection) — handler is unchanged.
The generic queue core lives in scaleoututil.queue; this adapter is the gRPC-specific
glue and lives next to GrpcHandler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import grpc
from google.protobuf.message import Message

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.grpc_handler import GRPC_PERMANENT_CODES, grpc_needs_reconnect
from scaleoututil.queue.interface import TransportAdapter
from scaleoututil.queue.record import (
    Acked,
    PermanentFailure,
    QueueRecord,
    SendResult,
    TransientFailure,
)

if TYPE_CHECKING:
    from scaleout.client.grpc_handler import GrpcHandler

# Per-message-type dispatch entry: (proto class, callable(handler, proto_instance) -> Any).
# Callable returns whatever the handler returns; raises on failure.
DispatchEntry = tuple[type[Message], Callable[["GrpcHandler", Message], Any]]


def _default_registry() -> dict[str, DispatchEntry]:
    return {
        "TelemetryMessage": (
            scaleout_msg.TelemetryMessage,
            lambda h, m: h.send_telemetry(m),
        ),
        "ModelMetric": (
            scaleout_msg.ModelMetric,
            lambda h, m: h.send_model_metric(m),
        ),
        "AttributeMessage": (
            scaleout_msg.AttributeMessage,
            lambda h, m: h.send_attributes(m),
        ),
        "AttributeRecord": (
            scaleout_msg.AttributeRecord,
            lambda h, m: h.send_attributes(m),
        ),
        "ModelUpdate": (
            scaleout_msg.ModelUpdate,
            lambda h, m: h._send_model_update(m),
        ),
        "ModelValidation": (
            scaleout_msg.ModelValidation,
            lambda h, m: h._send_model_validation(m),
        ),
        "Status": (
            scaleout_msg.Status,
            lambda h, m: h._send_status(m),
        ),
    }


class GrpcTransportAdapter(TransportAdapter):
    """Drains QueueRecords through a GrpcHandler.

    Responsibilities:
      - decode payload bytes into the right proto via message_type registry
      - dispatch to the matching handler method
      - translate handler return value / exception into SendResult
      - attempt a single channel reconnect on UNAVAILABLE before returning TransientFailure

    The queue drainer owns retry scheduling and backoff. The adapter makes one
    attempt per send() call; on a reconnectable failure it refreshes the channel
    (no sleep) so the next drain attempt has a fresh connection.
    """

    def __init__(
        self,
        handler: GrpcHandler,
        registry: Optional[dict[str, DispatchEntry]] = None,
    ) -> None:
        self._handler = handler
        self._registry = registry if registry is not None else _default_registry()

    def register(self, message_type: str, proto_class: type[Message], dispatch: Callable[[GrpcHandler, Message], Any]) -> None:
        """Add or override a dispatch entry. Useful for tests and future message types."""
        self._registry[message_type] = (proto_class, dispatch)

    def on_requeue(self, record_ids: list[str]) -> None:
        pass

    def send(self, record: QueueRecord) -> SendResult:
        entry = self._registry.get(record.message_type)
        if entry is None:
            return PermanentFailure(f"no dispatch entry for message_type={record.message_type!r}")

        proto_class, dispatch = entry
        try:
            proto = proto_class()
            proto.ParseFromString(record.payload)
        except Exception as e:
            return PermanentFailure(f"payload parse failed for {record.message_type}: {e!r}")

        try:
            dispatch(self._handler, proto)
            return Acked()
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            details = e.details() if hasattr(e, "details") else ""
            if code in GRPC_PERMANENT_CODES:
                return PermanentFailure(f"{code}: {details}")
            if grpc_needs_reconnect(e):
                try:
                    self._handler._reconnect_channel()
                except Exception:  # best-effort; TransientFailure below drives retry
                    pass
            return TransientFailure(f"{code}: {details}")
        except ValueError:
            try:
                self._handler._reconnect_channel()
            except Exception:  # best-effort; TransientFailure below drives retry
                pass
            return TransientFailure("ValueError: channel in bad state")
        except Exception as e:
            return TransientFailure(f"{type(e).__name__}: {e}")


def grpc_envelope(message: Message, priority: str) -> QueueRecord:
    """Build a QueueRecord from a protobuf message.

    The runtime calls this immediately before queue.enqueue(...). Producers never
    touch the queue's internal envelope shape; they pass live proto + priority and
    let the helper serialise + tag.
    """
    return QueueRecord.create(
        message_type=type(message).__name__,
        payload=message.SerializeToString(),
        priority=priority,
    )
