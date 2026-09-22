"""Unit tests for StreamingTelemetryRecordAdapter.

Mirrors test_streaming_telemetry.py with a fake stub that exposes the new
StreamTelemetryRecord RPC.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import List

import grpc

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.streaming_record import (
    BaseRoutingTransportAdapter,
    StreamingTelemetryRecordAdapter,
)
from scaleoututil.queue.record import SendResult
from scaleoututil.queue.record import (
    Acked,
    PermanentFailure,
    QueueRecord,
    Sent,
)


class _FakeRpcError(grpc.RpcError):
    def __init__(self, details: str = "broken") -> None:
        self._details = details

    def details(self) -> str:
        return self._details


class _FakeCall:
    def __init__(self, request_iterator, acks_to_yield: List[scaleout_msg.RecordAck], raise_on_iter: Exception = None):
        self._request_iterator = request_iterator
        self._acks = list(acks_to_yield)
        self._raise = raise_on_iter
        self._cancelled = threading.Event()
        self._received: list[scaleout_msg.TelemetryRecord] = []
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def _consume(self):
        for msg in self._request_iterator:
            self._received.append(msg)

    @property
    def received(self) -> list[scaleout_msg.TelemetryRecord]:
        return list(self._received)

    def cancel(self):
        self._cancelled.set()

    def __iter__(self):
        for ack in self._acks:
            if self._cancelled.is_set():
                return
            yield ack
        if self._raise is not None:
            raise self._raise


class _FakeStub:
    def __init__(self) -> None:
        self.last_call: _FakeCall = None
        self._acks: List[scaleout_msg.RecordAck] = []
        self._raise: Exception = None

    def queue_acks(self, ids: list[str]) -> None:
        self._acks.append(scaleout_msg.RecordAck(record_ids=ids))

    def script_raise(self, exc: Exception) -> None:
        self._raise = exc

    def StreamTelemetryRecord(self, request_iterator, metadata=None):
        call = _FakeCall(request_iterator, self._acks, raise_on_iter=self._raise)
        self.last_call = call
        self._acks = []
        self._raise = None
        return call


class _FakeHandler:
    def __init__(self) -> None:
        self.combinerStub = _FakeStub()
        self.metadata = [("auth", "token")]


def _record(rid: str = None) -> QueueRecord:
    proto = scaleout_msg.TelemetryRecord()
    proto.client_id = "client-1"
    proto.key = "cpu"
    proto.payload = '{"value": 0.5}'
    payload = proto.SerializeToString()
    record = QueueRecord.create(
        message_type="TelemetryRecord",
        payload=payload,
        priority="telemetry",
    )
    if rid is not None:
        from dataclasses import replace
        record = replace(record, record_id=rid)
    return record


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestStreamingTelemetryRecordAdapter(unittest.TestCase):
    def test_send_returns_sent_and_sets_telemetry_id(self):
        handler = _FakeHandler()
        adapter = StreamingTelemetryRecordAdapter(handler=handler, mark_acked=lambda _r: None)
        try:
            record = _record()
            result = adapter.send(record)
            self.assertIsInstance(result, Sent)
            self.assertTrue(_wait_for(lambda: len(handler.combinerStub.last_call.received) == 1))
            self.assertEqual(handler.combinerStub.last_call.received[0].telemetry_id, record.record_id)
        finally:
            adapter.close()

    def test_recv_loop_invokes_mark_acked_per_id(self):
        handler = _FakeHandler()
        acked: list[str] = []
        handler.combinerStub.queue_acks(["tid-a", "tid-b"])
        handler.combinerStub.queue_acks(["tid-c"])
        adapter = StreamingTelemetryRecordAdapter(handler=handler, mark_acked=acked.append)
        try:
            adapter.send(_record(rid="tid-a"))
            self.assertTrue(_wait_for(lambda: acked == ["tid-a", "tid-b", "tid-c"], timeout=3.0))
        finally:
            adapter.close()

    def test_rpc_error_marks_broken_and_reopens_on_next_send(self):
        handler = _FakeHandler()
        handler.combinerStub.script_raise(_FakeRpcError("boom"))
        adapter = StreamingTelemetryRecordAdapter(handler=handler, mark_acked=lambda _r: None)
        try:
            adapter.send(_record())
            self.assertTrue(_wait_for(lambda: adapter._broken, timeout=3.0))
            first_call = handler.combinerStub.last_call
            # Retry sends until backoff expires and the adapter reopens the stream
            self.assertTrue(_wait_for(
                lambda: (adapter.send(_record()), handler.combinerStub.last_call is not first_call)[1],
                timeout=3.0,
            ))
            self.assertFalse(adapter._broken)
        finally:
            adapter.close()

    def test_non_record_message_type_is_permanent_failure(self):
        handler = _FakeHandler()
        adapter = StreamingTelemetryRecordAdapter(handler=handler, mark_acked=lambda _r: None)
        try:
            record = QueueRecord.create(message_type="TelemetryMessage", payload=b"", priority="telemetry")
            result = adapter.send(record)
            self.assertIsInstance(result, PermanentFailure)
        finally:
            adapter.close()


class _RecordingAdapter:
    def __init__(self, result):
        self.sent: list[QueueRecord] = []
        self._result = result

    def send(self, record):
        self.sent.append(record)
        return self._result


class _TestRoutingAdapter(BaseRoutingTransportAdapter):
    """Minimal routing adapter for unit tests — takes pre-built sub-adapters."""

    def __init__(self, unary, streaming_telemetry=None, streaming_inference=None, streaming_attribute=None):
        super().__init__(mark_acked=lambda _: None, max_outstanding=256, requeue_inflight=lambda: None)
        self._unary = unary
        self._streaming_telemetry = streaming_telemetry
        self._streaming_inference = streaming_inference
        self._streaming_attribute = streaming_attribute

    def _do_send(self, record: QueueRecord) -> SendResult:
        if record.message_type == "TelemetryRecord" and self._streaming_telemetry is not None:
            return self._streaming_telemetry.send(record)
        if record.message_type == "InferenceResult" and self._streaming_inference is not None:
            return self._streaming_inference.send(record)
        if record.message_type == "AttributeRecord" and self._streaming_attribute is not None:
            return self._streaming_attribute.send(record)
        return self._unary.send(record)

    def close(self) -> None:
        pass


class TestRouting(unittest.TestCase):
    def test_record_routes_to_streaming_record(self):
        streaming_record = _RecordingAdapter(Sent())
        unary = _RecordingAdapter(Acked())
        adapter = _TestRoutingAdapter(unary=unary, streaming_telemetry=streaming_record)
        adapter.send(QueueRecord.create(message_type="TelemetryRecord", payload=b"", priority="telemetry"))
        self.assertEqual(len(streaming_record.sent), 1)
        self.assertEqual(len(unary.sent), 0)

    def test_legacy_telemetry_message_routes_to_unary(self):
        streaming_record = _RecordingAdapter(Sent())
        unary = _RecordingAdapter(Acked())
        adapter = _TestRoutingAdapter(unary=unary, streaming_telemetry=streaming_record)
        adapter.send(QueueRecord.create(message_type="TelemetryMessage", payload=b"", priority="telemetry"))
        self.assertEqual(len(unary.sent), 1)
        self.assertEqual(len(streaming_record.sent), 0)

    def test_record_falls_through_to_unary_when_no_streaming_record(self):
        unary = _RecordingAdapter(Acked())
        adapter = _TestRoutingAdapter(unary=unary)
        adapter.send(QueueRecord.create(message_type="TelemetryRecord", payload=b"", priority="telemetry"))
        self.assertEqual(len(unary.sent), 1)


if __name__ == "__main__":
    unittest.main()
