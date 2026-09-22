"""Streaming record adapters for the priority queue.

`BaseStreamingRecordAdapter` holds all bidi-stream lifecycle logic.
Concrete subclasses supply three small hooks (which RPC to call, how to
parse the proto, which field to stamp with the ack key).

`BaseRoutingTransportAdapter` is the ABC for the routing layer. It owns
the max-outstanding gate: `send()` reserves a slot, calls `_do_send()`,
and releases the slot if the result is not `Sent`. Acks decrement via
`mark_acked()`. The current outstanding count is exposed so the queue
can observe backlog pressure.

Legacy `TelemetryMessage` flows go through the unary `GrpcTransportAdapter`
(`handler.send_telemetry`) — there is no streaming legacy path.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Protocol

import grpc

from scaleout.client.grpc_handler import GrpcHandler, grpc_needs_reconnect
import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.queue.interface import TransportAdapter
from scaleoututil.queue.record import (
    PermanentFailure,
    QueueRecord,
    SendResult,
    Sent,
    TransientFailure,
)


_STOP: object = object()
_RECONNECT_BACKOFF_INITIAL_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 32.0
_DEFAULT_MAX_OUTSTANDING = 256

Metadata = list[tuple[str, str]]


class _StreamCall(Protocol):
    def cancel(self) -> bool: ...

    def __iter__(self) -> Iterator[scaleout_msg.RecordAck]: ...


# ---------------------------------------------------------------------------
# Routing layer ABC
# ---------------------------------------------------------------------------


class BaseRoutingTransportAdapter(ABC, TransportAdapter):
    """Flow-control base for routing adapters.

    Subclasses implement `_do_send` (routing logic) and `close`.
    This class owns the max-outstanding gate and exposes the current count.
    """

    def __init__(
        self,
        mark_acked: Callable[[str], None],
        max_outstanding: int = _DEFAULT_MAX_OUTSTANDING,
        requeue_inflight: Callable[[], None] = lambda: None,
    ) -> None:
        self._mark_acked_cb = mark_acked
        self._requeue_inflight = requeue_inflight
        self._max_outstanding = max_outstanding
        self._lock = threading.Lock()
        self._outstanding: int = 0

    @property
    def outstanding(self) -> int:
        with self._lock:
            return self._outstanding

    def send(self, record: QueueRecord) -> SendResult:
        with self._lock:
            if self._outstanding >= self._max_outstanding:
                return TransientFailure("too many unacked records")
            self._outstanding += 1
        result = self._do_send(record)
        if not isinstance(result, Sent):
            with self._lock:
                self._outstanding -= 1
        return result

    def mark_acked(self, record_id: str) -> None:
        with self._lock:
            self._outstanding = max(0, self._outstanding - 1)
        self._mark_acked_cb(record_id)

    def release_slots(self, n: int) -> None:
        """Release n outstanding slots and requeue in-flight records — called by streaming adapters on reconnect."""
        with self._lock:
            self._outstanding = max(0, self._outstanding - n)
        self._requeue_inflight()

    def on_requeue(self, record_ids: list[str]) -> None:
        """Release slots for records the backend TTL-requeued back to QUEUED."""
        with self._lock:
            self._outstanding = max(0, self._outstanding - len(record_ids))

    @abstractmethod
    def _do_send(self, record: QueueRecord) -> SendResult: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Streaming stream lifecycle
# ---------------------------------------------------------------------------


class BaseStreamingRecordAdapter(ABC):
    """Shared bidi-stream lifecycle: outbound queue, recv loop, reopen on error.

    Subclasses implement three hooks that differ per message type:
      - _open_stream: which RPC on the stub to call
      - _parse_proto: deserialise the queue payload bytes into a proto
      - _set_record_id: stamp the ack-correlation field on the proto
      - message_type: the string tag used for routing

    Backpressure (max-outstanding) lives in the routing layer, not here.
    `_adapter_outstanding` is a lightweight per-stream counter used solely to
    release the correct number of router slots when the stream reconnects.
    """

    def __init__(
        self,
        handler: GrpcHandler,
        mark_acked: Callable[[str], None],
        queue_size: int = _DEFAULT_MAX_OUTSTANDING,
        release_slots: Callable[[int], None] = lambda _: None,
    ) -> None:
        self._handler = handler
        self._mark_acked = mark_acked
        self._queue_size = queue_size
        self._release_slots = release_slots
        self._lock = threading.Lock()
        self._adapter_outstanding: int = 0
        self._outbound: Optional[queue.Queue] = None
        self._call: Optional[_StreamCall] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._broken = False
        self._closed = False
        self._reopen_after: float = 0.0
        self._backoff_s: float = _RECONNECT_BACKOFF_INITIAL_S

    @property
    @abstractmethod
    def message_type(self) -> str: ...

    @abstractmethod
    def _open_stream(self, request_iterator: Iterator) -> _StreamCall: ...

    @abstractmethod
    def _parse_proto(self, payload: bytes): ...

    @abstractmethod
    def _set_record_id(self, proto, record_id: str) -> None: ...

    # ---- TransportAdapter interface ------------------------------------------

    def send(self, record: QueueRecord) -> SendResult:
        if record.message_type != self.message_type:
            return PermanentFailure(f"{self.__class__.__name__} received unexpected message_type={record.message_type!r}")
        try:
            proto = self._parse_proto(record.payload)
        except Exception as e:
            return PermanentFailure(f"payload parse failed: {e!r}")
        self._set_record_id(proto, record.record_id)

        with self._lock:
            if self._closed:
                return TransientFailure("streaming adapter closed")
            if self._broken or self._outbound is None:
                if self._broken and time.monotonic() < self._reopen_after:
                    return TransientFailure("streaming adapter in backoff")
                self._reopen_locked()
            self._adapter_outstanding += 1
            outbound = self._outbound

        outbound.put(proto)
        return Sent()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            outbound = self._outbound
            call = self._call
            recv = self._recv_thread
            self._outbound = None
            self._call = None
            self._recv_thread = None

        if outbound is not None:
            try:
                outbound.put_nowait(_STOP)
            except queue.Full:  # call.cancel() below will kill the stream regardless
                pass
        if call is not None:
            try:
                call.cancel()
            except Exception as e:
                ScaleoutLogger().debug(f"streaming: cancel() during close raised (ignored): {e!r}")
        if recv is not None:
            recv.join(timeout=5.0)

    # ---- internals (caller holds self._lock for _reopen_locked) --------------

    def _reopen_locked(self) -> None:
        n_to_release = self._adapter_outstanding
        self._adapter_outstanding = 0

        if self._outbound is not None:
            try:
                self._outbound.put_nowait(_STOP)
            except queue.Full:  # call.cancel() below will kill the stream regardless
                pass
        if self._call is not None:
            try:
                self._call.cancel()
            except Exception as e:
                ScaleoutLogger().debug(f"streaming: cancel() during reopen raised (ignored): {e!r}")

        if n_to_release > 0:
            self._release_slots(n_to_release)

        outbound: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self._outbound = outbound
        try:
            self._call = self._open_stream(self._request_iterator(outbound))
        except grpc.RpcError as e:
            if grpc_needs_reconnect(e):
                try:
                    self._handler._reconnect_channel()
                except Exception:  # best-effort; original RpcError is re-raised regardless
                    pass
            raise
        self._broken = False
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            args=(self._call,),
            name=f"streaming-{self.message_type.lower()}-recv",
            daemon=True,
        )
        self._recv_thread.start()

    @staticmethod
    def _request_iterator(outbound: queue.Queue) -> Iterator:
        while True:
            item = outbound.get()
            if item is _STOP:
                return
            yield item

    def _recv_loop(self, call: _StreamCall) -> None:
        name = self.message_type.lower()
        try:
            for ack in call:
                for rid in ack.record_ids:
                    # Reset the backoff only after successfully recieving an ACK
                    self._backoff_s = _RECONNECT_BACKOFF_INITIAL_S
                    with self._lock:
                        self._adapter_outstanding = max(0, self._adapter_outstanding - 1)
                    try:
                        self._mark_acked(rid)
                    except Exception as e:
                        ScaleoutLogger().warning(f"streaming-{name}: mark_acked({rid}) raised: {e!r}")
        except grpc.RpcError as e:
            ScaleoutLogger().info(f"streaming-{name}: stream broken: {e!r}")
            with self._lock:
                self._reopen_after = time.monotonic() + self._backoff_s
                self._backoff_s = min(self._backoff_s * 2, _RECONNECT_BACKOFF_MAX_S)
        except Exception as e:
            ScaleoutLogger().warning(f"streaming-{name}: recv loop crashed: {e!r}")
        finally:
            with self._lock:
                self._broken = True


# ---------------------------------------------------------------------------
# Concrete streaming adapters
# ---------------------------------------------------------------------------


class StreamingTelemetryRecordAdapter(BaseStreamingRecordAdapter):
    """Bidi-streaming adapter for `TelemetryRecord` via `StreamTelemetryRecord`."""

    @property
    def message_type(self) -> str:
        return "TelemetryRecord"

    def _open_stream(self, request_iterator: Iterator) -> _StreamCall:
        return self._handler.combinerStub.StreamTelemetryRecord(
            request_iterator,
            metadata=self._handler.metadata,
        )

    def _parse_proto(self, payload: bytes) -> scaleout_msg.TelemetryRecord:
        proto = scaleout_msg.TelemetryRecord()
        proto.ParseFromString(payload)
        return proto

    def _set_record_id(self, proto: scaleout_msg.TelemetryRecord, record_id: str) -> None:
        proto.telemetry_id = record_id


class StreamingInferenceResultAdapter(BaseStreamingRecordAdapter):
    """Bidi-streaming adapter for `InferenceResult` via `StreamInferenceResult`."""

    @property
    def message_type(self) -> str:
        return "InferenceResult"

    def _open_stream(self, request_iterator: Iterator) -> _StreamCall:
        return self._handler.combinerStub.StreamInferenceResult(
            request_iterator,
            metadata=self._handler.metadata,
        )

    def _parse_proto(self, payload: bytes) -> scaleout_msg.InferenceResult:
        proto = scaleout_msg.InferenceResult()
        proto.ParseFromString(payload)
        return proto

    def _set_record_id(self, proto: scaleout_msg.InferenceResult, record_id: str) -> None:
        proto.inference_result_id = record_id


class StreamingAttributeRecordAdapter(BaseStreamingRecordAdapter):
    """Bidi-streaming adapter for `AttributeRecord` via `StreamAttributeRecord`."""

    @property
    def message_type(self) -> str:
        return "AttributeRecord"

    def _open_stream(self, request_iterator: Iterator) -> _StreamCall:
        return self._handler.combinerStub.StreamAttributeRecord(
            request_iterator,
            metadata=self._handler.metadata,
        )

    def _parse_proto(self, payload: bytes) -> scaleout_msg.AttributeRecord:
        proto = scaleout_msg.AttributeRecord()
        proto.ParseFromString(payload)
        return proto

    def _set_record_id(self, proto: scaleout_msg.AttributeRecord, record_id: str) -> None:
        proto.attribute_id = record_id


# ---------------------------------------------------------------------------
# Routing adapters
# ---------------------------------------------------------------------------


class GrpcRoutingTransportAdapter(BaseRoutingTransportAdapter):
    """Routing adapter that owns the lifecycle of its gRPC streaming adapters.

    Creates `StreamingTelemetryRecordAdapter`, `StreamingInferenceResultAdapter`,
    and `StreamingAttributeRecordAdapter` internally, wiring them to this
    router's `mark_acked` and `release_slots` callbacks.
    """

    def __init__(
        self,
        *,
        handler: GrpcHandler,
        unary: TransportAdapter,
        mark_acked: Callable[[str], None],
        requeue_inflight: Callable[[], None],
        use_legacy_telemetry: bool = False,
        max_outstanding: int = _DEFAULT_MAX_OUTSTANDING,
    ) -> None:
        super().__init__(mark_acked=mark_acked, max_outstanding=max_outstanding, requeue_inflight=requeue_inflight)
        self._unary = unary
        self._streaming_telemetry: Optional[StreamingTelemetryRecordAdapter] = None
        self._streaming_inference: Optional[StreamingInferenceResultAdapter] = None
        self._streaming_attribute = StreamingAttributeRecordAdapter(
            handler=handler,
            mark_acked=self.mark_acked,
            queue_size=max_outstanding,
            release_slots=self.release_slots,
        )
        if not use_legacy_telemetry:
            self._streaming_telemetry = StreamingTelemetryRecordAdapter(
                handler=handler,
                mark_acked=self.mark_acked,
                queue_size=max_outstanding,
                release_slots=self.release_slots,
            )
            self._streaming_inference = StreamingInferenceResultAdapter(
                handler=handler,
                mark_acked=self.mark_acked,
                queue_size=max_outstanding,
                release_slots=self.release_slots,
            )

    def _do_send(self, record: QueueRecord) -> SendResult:
        if record.message_type == "TelemetryRecord" and self._streaming_telemetry is not None:
            return self._streaming_telemetry.send(record)
        if record.message_type == "InferenceResult" and self._streaming_inference is not None:
            return self._streaming_inference.send(record)
        if record.message_type == "AttributeRecord":
            return self._streaming_attribute.send(record)
        return self._unary.send(record)

    def close(self) -> None:
        for adapter in [self._streaming_telemetry, self._streaming_inference, self._streaming_attribute]:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as e:
                    ScaleoutLogger().debug(f"streaming: close() raised (ignored): {e!r}")
