"""Integration tests for GrpcEdgeClientRuntime <-> PriorityQueue wiring (A.3).

Uses a fake GrpcHandler — no live combiner needed. Every test that constructs
a queue tears it down in teardown to avoid leaked drainer threads.
"""

from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import patch

import grpc

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.grpc_edge_client_runtime import (
    GrpcEdgeClientRuntime,
    _DEFAULT_PRIORITY_CLASSES,
)
from scaleout.client.grpc_handler import GrpcConnectionOptions
from scaleout.client.link_quality import LinkQualityEstimator
from scaleout.client.grpc_transport import GrpcTransportAdapter
from scaleoututil.queue import (
    DEFAULT_CLASS_NAME,
    InMemoryBackend,
    PriorityQueue,
)


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class _FakeHandler:
    """Records dispatched messages; satisfies the GrpcTransportAdapter contract.

    Each method appends to ``calls`` and returns ``True`` (or raises if scripted).
    """

    class _FakeChannel:
        def close(self):
            pass

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._script: dict[str, BaseException] = {}
        self.link_quality_estimator = LinkQualityEstimator()
        self.channel = self._FakeChannel()

    def script_raise(self, method: str, exc: BaseException) -> None:
        self._script[method] = exc

    def _dispatch(self, method: str, msg: Any) -> bool:
        self.calls.append((method, msg))
        exc = self._script.get(method)
        if exc is not None:
            raise exc
        return True

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

    def start_link_quality_monitor(self) -> None:
        pass

    def stop_link_quality_monitor(self, timeout=None) -> None:
        pass

    def check_version_compatibility(self):
        return True, "test-version", ""

    def disconnect(self):
        self.calls.append(("disconnect", None))
        return None

    def create_metric_message(
        self,
        metrics: dict,
        model_id: str,
        step: int,
        round_id: str,
        session_id: str,
    ) -> scaleout_msg.ModelMetric:
        metric = scaleout_msg.ModelMetric()
        metric.client_id = "test-client"
        metric.model_id = model_id
        if step is not None:
            metric.step.value = step
        metric.session_id = session_id
        metric.round_id = round_id
        metric.timestamp.GetCurrentTime()
        for key, value in metrics.items():
            metric.metrics.add(key=key, value=value)
        return metric


class _FakeClient:
    """Minimum surface used by GrpcEdgeClientRuntime."""

    def __init__(self, client_id: str = "test-client", name: str = "test-name") -> None:
        self.client_id = client_id
        self.name = name
        self.train_callback = None
        self.validate_callback = None
        self.stage_model_callback = None
        self.registered_callbacks = {}

    def set_client_id(self, client_id: str) -> None:
        self.client_id = client_id

    def set_name(self, name: str) -> None:
        self.name = name

    def log_telemetry(self, key=None, payload=None, telemetry=None, check_task_abort: bool = False) -> bool:
        return True

    def stop_default_telemetry_loop(self) -> None:
        pass

    @contextmanager
    def logging_context(self, _ctx_obj):
        yield


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` up to ``timeout`` seconds. Returns the last value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def _make_runtime_with_queue(handler: Optional[_FakeHandler] = None) -> tuple[GrpcEdgeClientRuntime, _FakeHandler]:
    """Build a runtime with a started queue wired to a fake handler — no gRPC.

    Uses the unary-only adapter, so telemetry has to go through the legacy
    `TelemetryMessage` -> `send_telemetry` dispatch entry (the new
    `TelemetryRecord` path has no unary RPC).
    """
    if handler is None:
        handler = _FakeHandler()
    client = _FakeClient()
    runtime = GrpcEdgeClientRuntime()
    runtime.set_client(client)
    runtime.grpc_handler = handler
    runtime._use_legacy_telemetry = True
    runtime._backend = InMemoryBackend(classes=_DEFAULT_PRIORITY_CLASSES)
    adapter = GrpcTransportAdapter(handler)
    runtime.queue = PriorityQueue(transport=adapter, backend=runtime._backend)
    runtime.queue.start()
    return runtime, handler


class TestInitGrpcHandlerConstructsQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime: Optional[GrpcEdgeClientRuntime] = None

    def tearDown(self) -> None:
        if self.runtime is not None and self.runtime.queue is not None:
            self.runtime.queue.stop(timeout=2.0)

    def test_init_grpchandler_constructs_queue(self):
        client = _FakeClient()
        fake_handler = _FakeHandler()
        with patch(
            "scaleout.client.grpc_edge_client_runtime.GrpcHandler",
            return_value=fake_handler,
        ):
            self.runtime = GrpcEdgeClientRuntime()
            self.runtime.set_client(client)
            ok = self.runtime.init_grpchandler(GrpcConnectionOptions(host="h", port=0))
        self.assertTrue(ok)
        self.assertIsNotNone(self.runtime.queue)
        self.assertIsNotNone(self.runtime.queue._thread)
        self.assertTrue(self.runtime.queue._thread.is_alive())

    def test_seven_classes_registered(self):
        client = _FakeClient()
        fake_handler = _FakeHandler()
        with patch(
            "scaleout.client.grpc_edge_client_runtime.GrpcHandler",
            return_value=fake_handler,
        ):
            self.runtime = GrpcEdgeClientRuntime()
            self.runtime.set_client(client)
            self.runtime.init_grpchandler(GrpcConnectionOptions(host="h", port=0))
        registered = set(self.runtime._backend.stats().keys())
        self.assertEqual(
            registered,
            {"alert", "model_update", "telemetry", "inference", "artifact", "backlog", DEFAULT_CLASS_NAME},
        )


class TestSendPathEnqueues(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.handler = _make_runtime_with_queue()

    def tearDown(self) -> None:
        self.runtime.queue.stop(timeout=2.0)

    def test_send_telemetry_enqueues_at_P5(self):
        # Back-compat positional dict — must still work.
        ok = self.runtime.send_telemetry({"cpu_usage": 12.0})
        self.assertTrue(ok)
        seen = _wait_until(lambda: any(c[0] == "send_telemetry" for c in self.handler.calls))
        self.assertTrue(seen)
        # Legacy mode emits one TelemetryMessage with all elems.
        msgs = [m for n, m in self.handler.calls if n == "send_telemetry"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual({e.key for e in msgs[0].telemetries}, {"cpu_usage"})

    def test_send_telemetry_new_key_form_in_legacy_mode_is_a_noop_for_metrics(self):
        # In legacy mode, `key=`/`payload=` cannot be translated into the
        # (key, float) wire shape; the call still succeeds but enqueues an
        # empty TelemetryMessage and logs a warning.
        ok = self.runtime.send_telemetry(key="loss", payload={"value": 0.25})
        self.assertTrue(ok)
        seen = _wait_until(lambda: any(c[0] == "send_telemetry" for c in self.handler.calls))
        self.assertTrue(seen)
        msgs = [m for n, m in self.handler.calls if n == "send_telemetry"]
        self.assertEqual(list(msgs[-1].telemetries), [])

    def test_send_metric_enqueues_at_P7(self):
        ok = self.runtime.send_metric(
            metrics={"loss": 0.5},
            model_id="m1",
            step=1,
            round_id="r1",
            session_id="s1",
        )
        self.assertTrue(ok)
        seen = _wait_until(lambda: any(c[0] == "send_model_metric" for c in self.handler.calls))
        self.assertTrue(seen)

    def test_send_attributes_enqueues_at_P7(self):
        ok = self.runtime.send_attributes({"label": "v"})
        self.assertTrue(ok)
        seen = _wait_until(lambda: any(c[0] == "send_attributes" for c in self.handler.calls))
        self.assertTrue(seen)

    def test_send_status_builds_status_proto(self):
        self.runtime.send_status(
            "hello",
            log_level=scaleout_msg.LogLevel.AUDIT,
            type="MODEL_UPDATE",
        )
        seen = _wait_until(lambda: any(c[0] == "_send_status" for c in self.handler.calls))
        self.assertTrue(seen)
        status_calls = [c for c in self.handler.calls if c[0] == "_send_status"]
        self.assertEqual(len(status_calls), 1)
        _, proto = status_calls[0]
        self.assertEqual(proto.client_id, "test-client")
        self.assertEqual(proto.log_level, scaleout_msg.LogLevel.AUDIT)
        self.assertEqual(proto.type, "MODEL_UPDATE")
        self.assertEqual(proto.status, "hello")


class TestUpdateLocalModelEnqueuesAtP4(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.handler = _make_runtime_with_queue()

        # Fakes for train_callback + get_model_from_combiner + send_model_to_combiner.
        class _FakeModel:
            model_id = "m-in"

            def to_builder(self):
                return self

            def set_model_id(self, mid):
                self.model_id = mid
                return self

            def build(self):
                return self

        self._fake_in = _FakeModel()
        self._fake_out = _FakeModel()
        self.runtime.get_model_from_combiner = lambda model_id: self._fake_in
        self.runtime.send_model_to_combiner = lambda model: None
        self.runtime._client.train_callback = lambda in_model, settings: (self._fake_out, {"training_metadata": {"num_examples": 10}})

    def tearDown(self) -> None:
        self.runtime.queue.stop(timeout=2.0)

    def test_update_local_model_enqueues_model_update_at_P4(self):
        req = scaleout_msg.TaskRequest()
        req.model_id = "m-in"
        req.correlation_id = "cid"
        req.round_id = "r1"
        req.session_id = "s1"
        req.data = "{}"
        self.runtime.update_local_model(req)
        seen = _wait_until(lambda: any(c[0] == "_send_model_update" for c in self.handler.calls))
        self.assertTrue(seen)

    def _make_req(self) -> scaleout_msg.TaskRequest:
        req = scaleout_msg.TaskRequest()
        req.model_id = "m-in"
        req.correlation_id = "cid"
        req.round_id = "r1"
        req.session_id = "s1"
        req.data = "{}"
        return req

    def test_zero_num_examples_raises(self):
        self.runtime._client.train_callback = lambda in_model, settings: (
            self._fake_out,
            {"training_metadata": {"num_examples": 0}},
        )
        with self.assertRaises(ValueError):
            self.runtime.update_local_model(self._make_req())

    def test_missing_num_examples_raises(self):
        self.runtime._client.train_callback = lambda in_model, settings: (
            self._fake_out,
            {"training_metadata": {}},
        )
        with self.assertRaises(ValueError):
            self.runtime.update_local_model(self._make_req())

    def test_negative_num_examples_raises(self):
        self.runtime._client.train_callback = lambda in_model, settings: (
            self._fake_out,
            {"training_metadata": {"num_examples": -1}},
        )
        with self.assertRaises(ValueError):
            self.runtime.update_local_model(self._make_req())


class TestValidateGlobalModelLogsCompletedUnconditionally(unittest.TestCase):
    def setUp(self) -> None:
        # Validation will fail permanently at the handler — adapter discards the
        # record, drainer moves on to the AUDIT status that the runtime enqueued
        # unconditionally after the validation.
        self.handler = _FakeHandler()
        self.handler.script_raise(
            "_send_model_validation",
            _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "bad"),
        )
        self.runtime, _ = _make_runtime_with_queue(self.handler)

        class _FakeModel:
            model_id = "m-in"

        self.runtime.get_model_from_combiner = lambda model_id: _FakeModel()
        self.runtime._client.validate_callback = lambda in_model: {"acc": 0.9}

    def tearDown(self) -> None:
        self.runtime.queue.stop(timeout=2.0)

    def test_validate_global_model_logs_completed_unconditionally(self):
        req = scaleout_msg.TaskRequest()
        req.model_id = "m-in"
        req.correlation_id = "cid"
        req.session_id = "s1"
        self.runtime.validate_global_model(req)

        # There are two status messages produced by validate_global_model when
        # status reporting is enabled: one "Processing..." and one "completed".
        # We assert that the AUDIT-level "completed" fires exactly once.
        def _completed_audit_seen() -> bool:
            for method, proto in self.handler.calls:
                if method != "_send_status":
                    continue
                if proto.log_level == scaleout_msg.LogLevel.AUDIT and "completed" in proto.status:
                    return True
            return False

        self.assertTrue(_wait_until(_completed_audit_seen))


class TestQueueStoppedOnRunExit(unittest.TestCase):
    def test_queue_stopped_on_polling_exit(self):
        runtime, _handler = _make_runtime_with_queue()
        queue_thread = runtime.queue._thread

        # Make _run_polling_client return promptly so run() exits its try block.
        class _StubTaskReceiver:
            def start(self):
                pass

            def wait_on_manager_thread(self):
                return None

            def has_current_tasks(self):
                return False

            def abort_all_current_tasks(self):
                pass

        runtime.task_receiver = _StubTaskReceiver()

        with patch(
            "scaleout.client.grpc_edge_client_runtime.SCALEOUT_GRACEFUL_CLIENT_CONNECTION",
            False,
        ), patch(
            "scaleout.client.grpc_edge_client_runtime.SCALEOUT_CLIENT_SEND_TELEMETRY",
            False,
        ):
            runtime.run(with_heartbeat=False, with_polling=True)

        self.assertIsNone(runtime.queue)
        queue_thread.join(timeout=2.0)
        self.assertFalse(queue_thread.is_alive())

    def test_queue_stopped_on_unexpected_exception(self):
        runtime, _handler = _make_runtime_with_queue()
        queue_thread = runtime.queue._thread

        class _BoomTaskReceiver:
            def start(self):
                raise RuntimeError("boom")

        runtime.task_receiver = _BoomTaskReceiver()

        with patch(
            "scaleout.client.grpc_edge_client_runtime.SCALEOUT_CLIENT_SEND_TELEMETRY",
            False,
        ):
            with self.assertRaises(RuntimeError):
                runtime.run(with_heartbeat=False, with_polling=True)

        self.assertIsNone(runtime.queue)
        queue_thread.join(timeout=2.0)
        self.assertFalse(queue_thread.is_alive())


class TestSendTelemetryRecordMode(unittest.TestCase):
    """send_telemetry in record mode emits one TelemetryRecord per (key, value)."""

    def setUp(self) -> None:
        self.runtime, self.handler = _make_runtime_with_queue()
        # Override the legacy default that _make_runtime_with_queue sets.
        self.runtime._use_legacy_telemetry = False

    def tearDown(self) -> None:
        self.runtime.queue.stop(timeout=2.0)

    def _capture_envelopes(self) -> list:
        captured: list = []
        original_enqueue = self.runtime.queue.enqueue

        def capture(envelope):
            captured.append(envelope)
            return original_enqueue(envelope)

        self.runtime.queue.enqueue = capture
        return captured

    def test_key_form_emits_one_record_with_payload(self):
        captured = self._capture_envelopes()
        self.runtime.send_telemetry(key="loss", payload={"value": 0.5, "step": 12})

        self.assertEqual(len(captured), 1)
        env = captured[0]
        self.assertEqual(env.message_type, "TelemetryRecord")
        proto = scaleout_msg.TelemetryRecord()
        proto.ParseFromString(env.payload)
        self.assertEqual(proto.key, "loss")
        import json as _json
        payload = _json.loads(proto.payload)
        self.assertEqual(payload, {"value": 0.5, "step": 12})

    def test_legacy_telemetry_dict_in_record_mode_emits_one_record_per_pair(self):
        captured = self._capture_envelopes()
        self.runtime.send_telemetry({"loss": 0.5, "acc": 0.9})

        records = []
        for env in captured:
            self.assertEqual(env.message_type, "TelemetryRecord")
            proto = scaleout_msg.TelemetryRecord()
            proto.ParseFromString(env.payload)
            records.append(proto)

        self.assertEqual(len(records), 2)
        import json as _json
        keys_to_value = {r.key: _json.loads(r.payload)["value"] for r in records}
        self.assertAlmostEqual(keys_to_value["loss"], 0.5, places=5)
        self.assertAlmostEqual(keys_to_value["acc"], 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
