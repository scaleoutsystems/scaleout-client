"""Unit tests for EdgeClient.start_autonomous_inference."""

import unittest

from scaleout.client.edge_client import EdgeClient
from scaleoututil.grpc.tasktype import TaskType


class _FakeRuntime:
    def __init__(self):
        self.calls = []
        self.current_task_id = "task-1"

    def set_client(self, client):
        self.client = client

    def get_current_task_id(self):
        return self.current_task_id

    def start_autonomous_task(self, callback, command_type, parameters):
        self.calls.append((callback, command_type, parameters))
        return None


class TestStartAutonomousInference(unittest.TestCase):
    def test_runs_with_no_params_without_raising(self):
        runtime = _FakeRuntime()
        client = EdgeClient(runtime=runtime)

        client.start_autonomous_inference("model-1")

        self.assertEqual(len(runtime.calls), 1)
        _, _, parameters = runtime.calls[0]
        self.assertEqual(parameters, {"model_id": "model-1"})

    def test_returns_task_and_passes_dict_params_and_result(self):
        runtime = _FakeRuntime()
        runtime.start_autonomous_task = lambda callback, command_type, parameters: runtime.calls.append(callback) or "task"
        client = EdgeClient(runtime=runtime)
        captured = {}

        def fake_run_inference(model, params):
            captured["params"] = params
            return {"result": 1}

        client.run_inference = fake_run_inference

        self.assertEqual(client.start_autonomous_inference("model-1"), "task")
        self.assertEqual(runtime.calls[0](), {"result": 1})
        self.assertEqual(captured["params"], {})

    def test_merges_model_id_into_provided_params(self):
        runtime = _FakeRuntime()
        client = EdgeClient(runtime=runtime)

        client.start_autonomous_inference("model-1", params={"threshold": 0.5})

        _, _, parameters = runtime.calls[0]
        self.assertEqual(parameters, {"threshold": 0.5, "model_id": "model-1"})

    def test_prefixes_command_type_with_autonomous(self):
        runtime = _FakeRuntime()
        client = EdgeClient(runtime=runtime)

        client.start_autonomous_inference("model-1")

        _, command_type, _ = runtime.calls[0]
        self.assertEqual(command_type, "Autonomous_" + TaskType.Inference.value)

    def test_run_inference_sees_the_same_context_as_an_externally_dispatched_task(self):
        """An externally-dispatched inference request runs under an InferenceContext
        keyed by the task's correlation_id (see GrpcEdgeClientRuntime._process_inference_request).
        An autonomous inference run should get the same treatment.
        """
        runtime = _FakeRuntime()
        runtime.current_task_id = "task-123"
        client = EdgeClient(runtime=runtime)

        captured = {}

        def fake_run_inference(model, params):
            captured["context"] = client.current_inference_context

        client.run_inference = fake_run_inference

        client.start_autonomous_inference("model-1")
        callback = runtime.calls[0][0]
        callback()

        context = captured["context"]
        self.assertIsNotNone(context)
        self.assertEqual(context.inference_id, "task-123")
        self.assertEqual(context.model_id, "model-1")
        self.assertIsNone(client.current_inference_context)


if __name__ == "__main__":
    unittest.main()
