"""Unit tests for EdgeClient.stage_model / run_inference / inference_context,
and the LocalModelRepository.stage_model building block they delegate to.

Uses a fake runtime — no live combiner needed.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scaleout.client.edge_client import EdgeClient
from scaleout.client.inference_context import InferenceContext
from scaleout.client.local_repository import LocalModelRepository
from scaleoututil.utils.model import ScaleoutModel


def _make_model(model_id=None) -> ScaleoutModel:
    model = ScaleoutModel.from_training_model([np.array([1.0, 2.0, 3.0])])
    if model_id is not None:
        model = model.to_builder().set_model_id(model_id).build()
    return model


class _FakeRuntime:
    """Minimal EdgeClientRuntime stub — only the surface stage_model/run_inference touch."""

    def __init__(self):
        self.client = None
        self.combiner_models = {}
        self.get_model_calls = []

    def set_client(self, client):
        self.client = client

    def get_model_from_combiner(self, model_id):
        self.get_model_calls.append(model_id)
        return self.combiner_models.get(model_id)


class EdgeClientStageModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime = _FakeRuntime()
        self.client = EdgeClient(runtime=self.runtime)
        self.client.local_repository = LocalModelRepository(repository_path=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stage_model_by_object_saves_to_local_repository_without_download(self):
        model = _make_model()

        staged = self.client.stage_model(model)

        self.assertIs(staged, model)
        self.assertTrue(self.client.local_repository.model_exists(model))
        self.assertEqual(self.runtime.get_model_calls, [])

    def test_stage_model_by_id_downloads_from_combiner_when_not_staged(self):
        model = _make_model(model_id="remote-model")
        self.runtime.combiner_models["remote-model"] = model

        staged = self.client.stage_model("remote-model")

        self.assertEqual(staged.model_id, "remote-model")
        self.assertEqual(self.runtime.get_model_calls, ["remote-model"])
        self.assertTrue(self.client.local_repository.model_exists("remote-model"))

    def test_stage_model_by_id_raises_when_combiner_has_no_such_model(self):
        with self.assertRaises(ValueError):
            self.client.stage_model("missing-model")

        self.assertEqual(self.runtime.get_model_calls, ["missing-model"])

    def test_stage_model_by_id_skips_download_when_already_staged(self):
        model = _make_model(model_id="cached-model")
        self.client.local_repository.stage_model(model)

        staged = self.client.stage_model("cached-model")

        self.assertEqual(staged.model_id, "cached-model")
        self.assertEqual(self.runtime.get_model_calls, [])

    def test_stage_model_is_idempotent_for_an_already_staged_object(self):
        model = _make_model()
        self.client.stage_model(model)

        # Staging the same model object again must not duplicate it or error.
        self.client.stage_model(model)

        matches = [m for m in self.client.local_repository.models if m.model_id == model.model_id]
        self.assertEqual(len(matches), 1)


class EdgeClientRunInferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime = _FakeRuntime()
        self.client = EdgeClient(runtime=self.runtime)
        self.client.local_repository = LocalModelRepository(repository_path=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_inference_raises_without_a_callback(self):
        model = _make_model()

        with self.assertRaisesRegex(ValueError, "No inference callback set"):
            self.client.run_inference(model=model, params={})

    def test_run_inference_invokes_callback_with_model_object_and_params(self):
        received = {}

        def callback(model, params):
            received["model"] = model
            received["params"] = params
            return "inference-result"

        self.client.set_inference_callback(callback)
        model = _make_model()

        result = self.client.run_inference(model=model, params={"threshold": 0.5})

        self.assertEqual(result, "inference-result")
        self.assertIs(received["model"], model)
        self.assertEqual(received["params"], {"threshold": 0.5})

    def test_run_inference_stages_model_by_id_before_calling_callback(self):
        model = _make_model(model_id="remote-model")
        self.runtime.combiner_models["remote-model"] = model
        received = {}
        self.client.set_inference_callback(lambda m, p: received.setdefault("model", m))

        self.client.run_inference(model="remote-model", params=None)

        self.assertEqual(received["model"].model_id, "remote-model")
        self.assertEqual(self.runtime.get_model_calls, ["remote-model"])

    def test_run_inference_raises_when_model_is_none(self):
        self.client.set_inference_callback(lambda m, p: m)

        with self.assertRaisesRegex(ValueError, "Model not found in repository"):
            self.client.run_inference(model=None, params={})

    def test_run_inference_propagates_stage_model_failure_for_unknown_id(self):
        self.client.set_inference_callback(lambda m, p: m)

        with self.assertRaises(ValueError):
            self.client.run_inference(model="missing-model", params={})


class EdgeClientInferenceContextTests(unittest.TestCase):
    def setUp(self):
        self.client = EdgeClient(runtime=_FakeRuntime())

    def test_inference_context_is_scoped_to_the_with_block(self):
        self.assertIsNone(self.client.current_inference_context)
        ctx = InferenceContext(inference_id="i-1", model_id="m-1")

        with self.client.inference_context(ctx):
            self.assertIs(self.client.current_inference_context, ctx)

        self.assertIsNone(self.client.current_inference_context)

    def test_inference_context_restores_previous_context_on_exit(self):
        outer = InferenceContext(inference_id="i-outer", model_id="m-outer")
        inner = InferenceContext(inference_id="i-inner", model_id="m-inner")

        with self.client.inference_context(outer):
            with self.client.inference_context(inner):
                self.assertIs(self.client.current_inference_context, inner)
            self.assertIs(self.client.current_inference_context, outer)


class LocalModelRepositoryStageModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.repo = LocalModelRepository(repository_path=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stage_model_persists_model_to_disk_and_cache(self):
        model = _make_model()

        self.repo.stage_model(model)

        self.assertTrue(self.repo.model_exists(model))
        self.assertTrue((Path(self.tmp_dir) / f"{model.model_id}.scm").exists())

    def test_stage_model_is_a_noop_when_model_already_exists(self):
        model = _make_model()
        self.repo.stage_model(model)

        # Staging again must not raise or create a duplicate cache entry.
        self.repo.stage_model(model)

        matches = [m for m in self.repo.models if m.model_id == model.model_id]
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
