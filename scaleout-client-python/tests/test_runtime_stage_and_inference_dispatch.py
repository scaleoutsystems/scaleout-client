"""Unit tests for GrpcEdgeClientRuntime._process_model_stage_request and
_process_inference_request — the combiner-triggered dispatch that wires an
incoming TaskRequest to EdgeClient.stage_model()/run_inference().

Uses a fake client — no live combiner or transport needed.
"""

import json
import unittest
from contextlib import contextmanager

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg

from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime
from scaleout.client.inference_context import InferenceContext


class _FakeClient:
    """Minimum surface used by the two dispatch methods under test."""

    def __init__(self):
        self.stage_model_calls = []
        self.stage_model_return = None
        self.stage_model_callback = None

        self.run_inference_calls = []
        self.run_inference_return = None
        self.inference_contexts = []

    def stage_model(self, model):
        self.stage_model_calls.append(model)
        return self.stage_model_return

    def run_inference(self, model, params):
        self.run_inference_calls.append((model, params))
        return self.run_inference_return

    @contextmanager
    def inference_context(self, context):
        self.inference_contexts.append(context)
        yield context


class ProcessModelStageRequestTests(unittest.TestCase):
    def setUp(self):
        self.runtime = GrpcEdgeClientRuntime()
        self.client = _FakeClient()
        self.runtime.set_client(self.client)

    def _request(self, model_id: str) -> scaleout_msg.TaskRequest:
        req = scaleout_msg.TaskRequest()
        req.model_id = model_id
        return req

    def test_stages_model_and_invokes_stage_model_callback(self):
        self.client.stage_model_return = "staged-model"
        received = []
        self.client.stage_model_callback = received.append

        self.runtime._process_model_stage_request(self._request("m-1"))

        self.assertEqual(self.client.stage_model_calls, ["m-1"])
        self.assertEqual(received, ["staged-model"])

    def test_does_not_require_a_stage_model_callback(self):
        self.client.stage_model_return = "staged-model"

        # Must not raise even though no stage_model_callback is registered.
        self.runtime._process_model_stage_request(self._request("m-2"))

        self.assertEqual(self.client.stage_model_calls, ["m-2"])

    def test_raises_when_model_id_is_missing(self):
        with self.assertRaisesRegex(ValueError, "Model ID is required to stage a model."):
            self.runtime._process_model_stage_request(self._request(""))

        self.assertEqual(self.client.stage_model_calls, [])


class ProcessInferenceRequestTests(unittest.TestCase):
    def setUp(self):
        self.runtime = GrpcEdgeClientRuntime()
        self.client = _FakeClient()
        self.runtime.set_client(self.client)

    def _request(self, model_id: str, data: dict, correlation_id: str = "") -> scaleout_msg.TaskRequest:
        req = scaleout_msg.TaskRequest()
        req.model_id = model_id
        req.data = json.dumps(data)
        req.correlation_id = correlation_id
        return req

    def test_runs_inference_with_parsed_parameters_and_scoped_context(self):
        self.client.run_inference_return = "inference-result"
        request = self._request("m-1", {"parameters": {"threshold": 0.9}}, correlation_id="corr-1")

        result = self.runtime._process_inference_request(request)

        self.assertEqual(result, "inference-result")
        self.assertEqual(self.client.run_inference_calls, [("m-1", {"threshold": 0.9})])
        self.assertEqual(len(self.client.inference_contexts), 1)
        ctx = self.client.inference_contexts[0]
        self.assertIsInstance(ctx, InferenceContext)
        self.assertEqual(ctx.inference_id, "corr-1")
        self.assertEqual(ctx.model_id, "m-1")

    def test_defaults_to_empty_parameters_when_data_has_none(self):
        request = self._request("m-1", {}, correlation_id="corr-2")

        self.runtime._process_inference_request(request)

        self.assertEqual(self.client.run_inference_calls, [("m-1", {})])

    def test_raises_when_model_id_is_missing(self):
        request = self._request("", {"parameters": {}})

        with self.assertRaisesRegex(ValueError, "Model ID is required to run inference."):
            self.runtime._process_inference_request(request)

        self.assertEqual(self.client.run_inference_calls, [])


if __name__ == "__main__":
    unittest.main()
