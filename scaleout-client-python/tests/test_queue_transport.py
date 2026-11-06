"""Unit tests for GrpcTransportAdapter and grpc_envelope.

Uses a fake GrpcHandler — no live combiner needed.
"""

from __future__ import annotations

import unittest
from typing import Any

import grpc

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.grpc_transport import GrpcTransportAdapter, grpc_envelope
from scaleoututil.queue.record import Acked, PermanentFailure, TransientFailure


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class _FakeHandler:
    """Minimal stand-in for GrpcHandler.

    Records every call. Behaviour per method can be scripted to return a value or raise.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._script: dict[str, Any] = {}

    def script(self, method: str, behaviour: Any) -> None:
        """behaviour: a return value, or an Exception instance to raise."""
        self._script[method] = behaviour

    def _dispatch(self, method: str, msg: Any) -> Any:
        self.calls.append((method, msg))
        behaviour = self._script.get(method)
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour if behaviour is not None else True

    def send_telemetry(self, msg):
        return self._dispatch("send_telemetry", msg)

    def send_model_metric(self, msg):
        return self._dispatch("send_model_metric", msg)

    def send_attributes(self, msg):
        return self._dispatch("send_attributes", msg)

    def _send_model_update(self, msg):
        return self._dispatch("_send_model_update", msg)

    def _send_model_validation(self, msg):
        return self._dispatch("_send_model_validation", msg)

    def _send_status(self, msg):
        return self._dispatch("_send_status", msg)


class TestGrpcEnvelope(unittest.TestCase):
    def test_envelope_round_trip(self):
        telemetry = scaleout_msg.TelemetryMessage()
        telemetry.client_id = "client-a"
        telemetry.timestamp.GetCurrentTime()

        record = grpc_envelope(telemetry, priority="telemetry")

        self.assertEqual(record.message_type, "TelemetryMessage")
        self.assertEqual(record.priority, "telemetry")
        self.assertTrue(record.record_id)
        # payload must decode back to an equivalent proto
        decoded = scaleout_msg.TelemetryMessage()
        decoded.ParseFromString(record.payload)
        self.assertEqual(decoded.client_id, "client-a")


class TestGrpcAdapterDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = _FakeHandler()
        self.adapter = GrpcTransportAdapter(self.handler)

    def test_telemetry_dispatches_to_send_telemetry(self):
        msg = scaleout_msg.TelemetryMessage()
        msg.client_id = "c1"
        record = grpc_envelope(msg, priority="telemetry")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(len(self.handler.calls), 1)
        method, proto = self.handler.calls[0]
        self.assertEqual(method, "send_telemetry")
        self.assertEqual(proto.client_id, "c1")

    def test_model_metric_dispatch(self):
        msg = scaleout_msg.ModelMetric()
        msg.client_id = "c1"
        record = grpc_envelope(msg, priority="artifact")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(self.handler.calls[0][0], "send_model_metric")

    def test_attributes_dispatch(self):
        msg = scaleout_msg.AttributeMessage()
        msg.client_id = "c1"
        record = grpc_envelope(msg, priority="artifact")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(self.handler.calls[0][0], "send_attributes")

    def test_model_update_dispatch(self):
        msg = scaleout_msg.ModelUpdate()
        msg.client_id = "c1"
        msg.model_update_id = "u1"
        record = grpc_envelope(msg, priority="model_update")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(self.handler.calls[0][0], "_send_model_update")

    def test_model_validation_dispatch(self):
        msg = scaleout_msg.ModelValidation()
        msg.client_id = "c1"
        record = grpc_envelope(msg, priority="artifact")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(self.handler.calls[0][0], "_send_model_validation")

    def test_status_dispatch(self):
        msg = scaleout_msg.Status()
        msg.client_id = "c1"
        msg.status = "starting"
        record = grpc_envelope(msg, priority="alert")

        result = self.adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(self.handler.calls[0][0], "_send_status")

    def test_unknown_message_type_is_permanent_failure(self):
        record = grpc_envelope(scaleout_msg.TelemetryMessage(), priority="telemetry")
        # override message_type to something unregistered
        from dataclasses import replace
        record = replace(record, message_type="NopeMessage")
        result = self.adapter.send(record)
        self.assertIsInstance(result, PermanentFailure)


class TestGrpcAdapterErrorTranslation(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = _FakeHandler()
        self.adapter = GrpcTransportAdapter(self.handler)
        self.msg = scaleout_msg.TelemetryMessage()
        self.msg.client_id = "c1"
        self.record = grpc_envelope(self.msg, priority="telemetry")

    def test_unavailable_is_transient(self):
        self.handler.script("send_telemetry", _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no route"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, TransientFailure)

    def test_deadline_exceeded_is_transient(self):
        self.handler.script("send_telemetry", _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "slow"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, TransientFailure)

    def test_invalid_argument_is_permanent(self):
        self.handler.script("send_telemetry", _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "bad shape"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, PermanentFailure)

    def test_unauthenticated_is_permanent(self):
        self.handler.script("send_telemetry", _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED, "no token"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, PermanentFailure)

    def test_retry_exception_treated_as_transient(self):
        # @grpc_retry raises RetryException after exhausting attempts; we simulate the equivalent
        class RetryException(Exception):
            pass

        self.handler.script("send_telemetry", RetryException("Max retries exceeded"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, TransientFailure)

    def test_unexpected_exception_treated_as_transient(self):
        # conservative: don't lose records on unexpected handler bugs
        self.handler.script("send_telemetry", RuntimeError("boom"))
        result = self.adapter.send(self.record)
        self.assertIsInstance(result, TransientFailure)


class TestCustomRegistry(unittest.TestCase):
    def test_register_extra_message_type(self):
        handler = _FakeHandler()
        adapter = GrpcTransportAdapter(handler)
        captured = []
        adapter.register(
            "CustomMsg",
            scaleout_msg.Status,  # any proto type — reused for the parse step
            lambda h, m: captured.append(m),
        )
        msg = scaleout_msg.Status()
        msg.client_id = "c1"
        record = grpc_envelope(msg, priority="alert")
        from dataclasses import replace
        record = replace(record, message_type="CustomMsg")

        result = adapter.send(record)

        self.assertIsInstance(result, Acked)
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
