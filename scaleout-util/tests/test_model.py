import io
import os
import tempfile
import zipfile
import pytest
import numpy as np

from scaleoututil.utils.model import ScaleoutModel, ScaleoutModelBuilder
from scaleoututil.helpers.plugins.numpyhelper import Helper


@pytest.fixture
def helper():
    return Helper()


@pytest.fixture
def params():
    return [np.array([1.0, 2.0, 3.0]), np.array([[4.0, 5.0], [6.0, 7.0]])]


@pytest.fixture
def model(helper, params):
    """Model built from params."""
    return (
        ScaleoutModelBuilder.from_training_model(params, helper)
        .set_metadata("session_id", "s1")
        .build()
    )


@pytest.fixture
def file_model(helper, params):
    """Model saved to disk and reloaded."""
    m = ScaleoutModelBuilder.from_training_model(params, helper).build()
    with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
        path = f.name
    m.save_to_file(path)
    return ScaleoutModelBuilder.from_file(path).build()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def json_from_zip(zf):
    import json
    return json.loads(zf.read("metadata.json").decode())


class TestMetadata:
    def test_metadata_preloaded_after_construction(self, model):
        assert model.metadata["session_id"] == "s1"
        assert model.metadata["helper_type"] == "numpyhelper"

    def test_metadata_returns_copy(self, model):
        copy = model.metadata
        copy["injected"] = "bad"
        assert "injected" not in model.metadata


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

class TestModelParams:
    def test_has_training_model_true_after_construction(self, model):
        assert model.has_training_model

    def test_get_training_model_returns_correct_values(self, model, helper, params):
        loaded = model.get_training_model(helper)
        assert len(loaded) == len(params)
        for a, b in zip(loaded, params):
            assert np.allclose(a, b)

    def test_get_training_model_no_helper_raises(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        m._helper = None
        with pytest.raises(ValueError, match="No helper"):
            m.get_training_model()

    def test_from_training_model_zip_path_set(self, model):
        assert model._zip_path is not None
        assert os.path.exists(model._zip_path)


# ---------------------------------------------------------------------------
# Alternative representation
# ---------------------------------------------------------------------------

class TestFullModelRepr:
    def test_has_inference_model_false_by_default(self, model):
        assert not model.has_inference_model

    def test_has_inference_model_true_when_set_via_factory(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(b"data", "onnx").build()
        assert m.has_inference_model

    def test_get_inference_model_format_returns_none_by_default(self, model):
        assert model.get_inference_model_format() is None

    def test_get_inference_model_format_returns_format_string(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(b"data", "pt").build()
        assert m.get_inference_model_format() == "pt"

    def test_get_inference_model_stream_yields_correct_bytes(self, helper, params):
        fake_onnx = b"\x08\x07fake-onnx-payload"
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(fake_onnx, "onnx").build()
        with m.get_inference_model_stream() as s:
            data = s.read()
        assert data == fake_onnx

    def test_set_inference_model_updates_inference_model_format_metadata(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(b"data", "pt").build()
        assert m.metadata["inference_model_format"] == "pt"

    def test_set_inference_model_raises_if_not_bytes(self, helper, params):
        with pytest.raises(TypeError):
            ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model("not-bytes", "onnx")

    def test_full_model_survives_roundtrip(self, helper, params):
        full_model_data = b"\x00\x01\x02\x03fake-pt-data"
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(full_model_data, "pt").build()
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        with loaded.get_inference_model_stream() as s:
            data = s.read()
        assert data == full_model_data
        assert loaded.get_inference_model_format() == "pt"
        assert loaded.metadata["inference_model_format"] == "pt"

    def test_no_full_model_survives_roundtrip(self, model):
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        model.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert not loaded.has_inference_model
        assert loaded.get_inference_model_format() is None

    def test_full_model_lazy_from_zip(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(b"fake-onnx", "onnx").build()
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert loaded.has_inference_model
        with loaded.get_inference_model_stream() as s:
            data = s.read()
        assert data == b"fake-onnx"
        assert loaded.get_inference_model_format() == "onnx"


# ---------------------------------------------------------------------------
# Save / load round-trip (ZIP format)
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_metadata_survives_roundtrip(self, helper, params):
        m = (
            ScaleoutModelBuilder.from_training_model(params, helper)
            .set_metadata("session_id", "s1")
            .set_metadata("num_examples", 200)
            .build()
        )
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert loaded.metadata["session_id"] == "s1"
        assert loaded.metadata["num_examples"] == 200
        assert loaded.metadata["helper_type"] == "numpyhelper"

    def test_training_model_survive_roundtrip(self, model, helper, params):
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        model.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        result = loaded.get_training_model(helper)
        for a, b in zip(result, params):
            assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Legacy backward compatibility
# ---------------------------------------------------------------------------

class TestLegacyBackwardCompat:
    def test_legacy_stream_loads_without_error(self, helper, params):
        legacy = ScaleoutModel.from_training_model(params, helper)
        with legacy.get_training_model_stream() as raw_stream:
            loaded = ScaleoutModel.from_stream(raw_stream)
        assert isinstance(loaded, ScaleoutModel)

    def test_legacy_stream_gets_generated_model_id(self, helper, params):
        legacy = ScaleoutModel.from_training_model(params, helper)
        with legacy.get_training_model_stream() as s:
            raw_bytes = s.read()
        loaded = ScaleoutModel.from_stream(io.BytesIO(raw_bytes))
        assert loaded.model_id is not None

    def test_legacy_training_model_accessible_with_helper(self, helper, params):
        legacy = ScaleoutModel.from_training_model(params, helper)
        with legacy.get_training_model_stream() as s:
            raw_bytes = s.read()
        loaded = ScaleoutModel.from_stream(io.BytesIO(raw_bytes))
        result = loaded.get_training_model(helper)
        for a, b in zip(result, params):
            assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Existing API unchanged: checksum, get_file_stream, get_filechunk_stream
# ---------------------------------------------------------------------------

class TestExistingApiUnchanged:
    def test_checksum_is_string(self, model):
        cs = model.checksum
        assert isinstance(cs, str) and len(cs) > 0

    def test_verify_checksum_passes(self, model):
        assert model.verify_checksum(model.checksum)

    def test_verify_checksum_none_passes(self, model):
        assert model.verify_checksum(None)

    def test_verify_checksum_wrong_fails(self, model):
        assert not model.verify_checksum("badhash")

    def test_get_file_stream_returns_correct_bytes(self, model):
        s1 = model.get_file_stream()
        s2 = model.get_file_stream()
        assert s1.read() == s2.read()
        s1.close()
        s2.close()

    def test_get_filechunk_stream_yields_chunks(self, model):
        chunks = list(model.get_filechunk_stream())
        assert len(chunks) > 0
        total = b"".join(c.data for c in chunks)
        assert len(total) > 0

    def test_from_filechunk_stream_roundtrip(self, model, helper, params):
        chunks = model.get_filechunk_stream()
        rebuilt = ScaleoutModel.from_filechunk_stream(chunks)
        result = rebuilt.get_training_model(helper)
        for a, b in zip(result, params):
            assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# ZIP caching and ownership
# ---------------------------------------------------------------------------

class TestZipCaching:
    def test_from_file_does_not_own_zip(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert loaded._zip_is_owned is False

    def test_from_file_zip_path_equals_original(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert os.path.abspath(loaded._zip_path) == os.path.abspath(path)

    def test_from_file_does_not_delete_source_on_gc(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        del loaded
        assert os.path.exists(path)

    def test_from_file_keeps_file_handle_open(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        assert loaded._file_handle is not None
        assert not loaded._file_handle.closed

    def test_file_handle_closed_after_cleanup(self, helper, params):
        with ScaleoutModel.from_training_model(params, helper) as m:
            fh = m._file_handle
            assert not fh.closed
        assert fh.closed

    def test_from_training_model_keeps_file_handle_open(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        assert m._file_handle is not None
        assert not m._file_handle.closed

    def test_from_stream_zip_creates_owned_temp(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        buf = io.BytesIO()
        with open(m._zip_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        loaded = ScaleoutModel.from_stream(buf)
        assert loaded._zip_path is not None
        assert loaded._zip_is_owned is True

    def test_get_file_stream_returns_valid_zip(self, model):
        stream = model.get_file_stream()
        data = stream.read()
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            assert "metadata.json" in zf.namelist()
            assert "training_model.bin" in zf.namelist()

    def test_save_to_file_updates_zip_reference(self, model):
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        model.save_to_file(path)
        assert os.path.abspath(model._zip_path) == os.path.abspath(path)
        assert model._zip_is_owned is False

    def test_get_training_model_stream_reads_directly_from_zip(self, helper, params):
        m = ScaleoutModel.from_training_model(params, helper)
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        m.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        # Stream reads directly from ZIP — no temp file created
        with loaded.get_training_model_stream() as s:
            data = s.read()
        assert len(data) > 0
        assert not hasattr(loaded, "_data_path")

    def test_save_to_file_idempotent(self, model):
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        model.save_to_file(path)
        mtime1 = os.path.getmtime(path)
        model.save_to_file(path)
        mtime2 = os.path.getmtime(path)
        assert mtime1 == mtime2


# ---------------------------------------------------------------------------
# __repr__ and context manager
# ---------------------------------------------------------------------------

class TestReprAndContextManager:
    def test_repr_contains_model_id(self, model):
        r = repr(model)
        assert "numpyhelper" in r
        assert "ScaleoutModel" in r

    def test_context_manager_cleans_up(self, helper, params):
        with ScaleoutModel.from_training_model(params, helper) as m:
            zip_path = m._zip_path
            assert os.path.exists(zip_path)
        assert not os.path.exists(zip_path)


# ---------------------------------------------------------------------------
# ScaleoutModelBuilder — builder pattern
# ---------------------------------------------------------------------------

class TestFactory:
    def test_from_training_model_build_returns_correct_params(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).build()
        loaded = m.get_training_model(helper)
        for a, b in zip(loaded, params):
            assert np.allclose(a, b)

    def test_set_metadata_reflected_in_built_model(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_metadata("session_id", "s99").build()
        assert m.metadata["session_id"] == "s99"

    def test_set_metadata_dict_reflected_in_built_model(self, helper, params):
        m = (
            ScaleoutModelBuilder.from_training_model(params, helper)
            .set_metadata_dict({"a": 1, "b": 2})
            .build()
        )
        assert m.metadata["a"] == 1
        assert m.metadata["b"] == 2

    def test_set_inference_model_reflected_in_built_model(self, helper, params):
        full_model_data = b"\x08\x07fake-onnx"
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model(full_model_data, "onnx").build()
        with m.get_inference_model_stream() as s:
            result = s.read()
        assert result == full_model_data
        assert m.get_inference_model_format() == "onnx"

    def test_from_training_model_always_generates_new_model_id(self, helper, params):
        m = (
            ScaleoutModelBuilder.from_training_model(params, helper)
            .set_metadata("model_id", "fixed-id")
            .build()
        )
        assert m.model_id != "fixed-id"
        assert m.model_id is not None

    def test_from_training_model_each_build_has_unique_model_id(self, helper, params):
        m1 = ScaleoutModelBuilder.from_training_model(params, helper).build()
        m2 = ScaleoutModelBuilder.from_training_model(params, helper).build()
        assert m1.model_id != m2.model_id

    def test_from_file_preserves_model_id(self, helper, params):
        original = ScaleoutModelBuilder.from_training_model(params, helper).build()
        original_id = original.model_id
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        original.save_to_file(path)
        loaded = ScaleoutModelBuilder.from_file(path).build()
        assert loaded.model_id == original_id

    def test_from_file_with_set_inference_model_generates_new_model_id(self, helper, params):
        original = ScaleoutModelBuilder.from_training_model(params, helper).build()
        original_id = original.model_id
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        original.save_to_file(path)
        loaded = ScaleoutModelBuilder.from_file(path).set_inference_model(b"onnx-data", "onnx").build()
        assert loaded.model_id != original_id

    def test_from_stream_preserves_model_id(self, helper, params):
        original = ScaleoutModelBuilder.from_training_model(params, helper).build()
        original_id = original.model_id
        buf = io.BytesIO()
        with open(original._zip_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        loaded = ScaleoutModelBuilder.from_stream(buf).build()
        assert loaded.model_id == original_id

    def test_chaining_returns_self(self, helper, params):
        factory = ScaleoutModelBuilder.from_training_model(params, helper)
        assert factory.set_metadata("k", "v") is factory
        assert factory.set_metadata_dict({"x": 1}) is factory
        assert factory.set_inference_model(b"data", "onnx") is factory
        assert factory.set_model_id("m-001") is factory

    def test_set_inference_model_raises_if_not_bytes(self, helper, params):
        with pytest.raises(TypeError):
            ScaleoutModelBuilder.from_training_model(params, helper).set_inference_model("not-bytes", "onnx")

    def test_from_filechunk_stream_roundtrip(self, helper, params):
        original = ScaleoutModelBuilder.from_training_model(params, helper).build()
        chunks = original.get_filechunk_stream()
        loaded = ScaleoutModelBuilder.from_filechunk_stream(chunks).build()
        result = loaded.get_training_model(helper)
        for a, b in zip(result, params):
            assert np.allclose(a, b)

    def test_set_model_id_on_load_path(self, helper, params):
        original = ScaleoutModelBuilder.from_training_model(params, helper).build()
        with tempfile.NamedTemporaryFile(suffix=".scm", delete=False) as f:
            path = f.name
        original.save_to_file(path)
        loaded = ScaleoutModelBuilder.from_file(path).set_model_id("custom-id").build()
        assert loaded.model_id == "custom-id"

    def test_set_model_id_ignored_on_new_model_path(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).set_model_id("should-be-ignored").build()
        assert m.model_id != "should-be-ignored"
        assert m.model_id is not None


# ---------------------------------------------------------------------------
# to_builder — copy with modifications
# ---------------------------------------------------------------------------

class TestToBuilder:
    def test_to_builder_returns_factory(self, model):
        assert isinstance(model.to_builder(), ScaleoutModelBuilder)

    def test_to_builder_preserves_training_model(self, model, helper, params):
        copy = model.to_builder().build()
        result = copy.get_training_model(helper)
        for a, b in zip(result, params):
            assert np.allclose(a, b)

    def test_to_builder_preserves_metadata(self, model):
        copy = model.to_builder().build()
        assert copy.metadata["session_id"] == "s1"
        assert copy.metadata["helper_type"] == "numpyhelper"

    def test_to_builder_set_model_id_overrides(self, model):
        copy = model.to_builder().set_model_id("new-id").build()
        assert copy.model_id == "new-id"

    def test_to_builder_does_not_mutate_original(self, model):
        original_id = model.model_id
        model.to_builder().set_model_id("new-id").build()
        assert model.model_id == original_id

    def test_to_builder_set_metadata_adds_key(self, model):
        copy = model.to_builder().set_metadata("extra", 42).build()
        assert copy.metadata["extra"] == 42
        assert "extra" not in model.metadata

    def test_to_builder_set_inference_model_generates_new_model_id(self, model):
        copy = model.to_builder().set_inference_model(b"onnx-data", "onnx").build()
        assert copy.model_id != model.model_id
        with copy.get_inference_model_stream() as s:
            full = s.read()
        assert full == b"onnx-data"
        assert copy.get_inference_model_format() == "onnx"

    def test_to_builder_on_file_model(self, file_model):
        copy = file_model.to_builder().set_model_id("from-file-copy").build()
        assert copy.model_id == "from-file-copy"

    def test_model_id_setter_removed(self, model):
        """model_id setter no longer exists — only readable via property."""
        with pytest.raises(AttributeError):
            model.model_id = "should-fail"


# ---------------------------------------------------------------------------
# Helper type
# ---------------------------------------------------------------------------

class TestHelperType:
    def test_helper_stored_as_helper_type_in_metadata(self, model):
        assert "helper_type" in model.metadata
        assert model.metadata["helper_type"] == "numpyhelper"

    def test_set_helper_type_builds_with_correct_helper(self, params):
        m = ScaleoutModelBuilder.from_training_model(params).set_helper_type("numpyhelper").build()
        assert m._helper is not None
        assert m._helper.name == "numpyhelper"

    def test_helper_auto_loaded_from_metadata_on_file_load(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).build()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            path = f.name
        try:
            m.save_to_file(path)
            loaded = ScaleoutModelBuilder.from_file(path).build()
            assert loaded._helper is not None
            assert loaded._helper.name == "numpyhelper"
        finally:
            os.unlink(path)

    def test_helper_auto_loaded_from_metadata_on_stream_load(self, helper, params):
        m = ScaleoutModelBuilder.from_training_model(params, helper).build()
        with m.get_file_stream() as f:
            loaded = ScaleoutModelBuilder.from_stream(f).build()
        assert loaded._helper is not None
        assert loaded._helper.name == "numpyhelper"


class TestSigning:
    """Cryptographic signing via Ed25519."""

    @pytest.fixture
    def key_pair(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key = Ed25519PrivateKey.generate()
        return private_key, private_key.public_key()

    def test_sign_returns_dict_with_required_keys(self, params, key_pair):
        private_key, _ = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key)
        assert sig["model_id"] == model.model_id
        assert sig["algorithm"] == "ed25519"
        assert "signature" in sig

    def test_sign_with_signer_id(self, params, key_pair):
        private_key, _ = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key, signer_id="combiner-1")
        assert sig["signer_id"] == "combiner-1"

    def test_sign_without_signer_id(self, params, key_pair):
        private_key, _ = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key)
        assert "signer_id" not in sig

    def test_verify_signature_correct_key(self, params, key_pair):
        private_key, public_key = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key)
        assert model.verify_signature(public_key, sig["signature"]) is True

    def test_verify_signature_wrong_key(self, params, key_pair):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key, _ = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key)
        wrong_key = Ed25519PrivateKey.generate().public_key()
        assert model.verify_signature(wrong_key, sig["signature"]) is False

    def test_verify_signature_invalid_string(self, params, key_pair):
        _, public_key = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        assert model.verify_signature(public_key, "not-a-valid-signature") is False

    def test_sign_does_not_mutate_model(self, params, key_pair):
        private_key, _ = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        original_id = model.model_id
        model.sign(private_key)
        assert model.model_id == original_id

    def test_sign_on_file_model(self, params, key_pair, tmp_path):
        private_key, public_key = key_pair
        original = ScaleoutModelBuilder.from_training_model(params).build()
        path = str(tmp_path / "model.scm")
        original.save_to_file(path)
        loaded = ScaleoutModel.from_file(path)
        sig = loaded.sign(private_key)
        assert sig["model_id"] == original.model_id
        assert loaded.verify_signature(public_key, sig["signature"]) is True

    def test_modifying_model_invalidates_old_signature(self, params, key_pair):
        """Content change produces a new ZIP — signature over old content must not verify."""
        private_key, public_key = key_pair
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig = model.sign(private_key)
        new_params = [np.array([9.0, 8.0, 7.0])]
        modified = model.to_builder().set_training_model(new_params).build()
        assert modified.verify_signature(public_key, sig["signature"]) is False

    def test_multiple_signatures_each_verify_independently(self, params):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key_a = Ed25519PrivateKey.generate()
        key_b = Ed25519PrivateKey.generate()
        model = ScaleoutModelBuilder.from_training_model(params).build()
        sig_a = model.sign(key_a, signer_id="node-a")
        sig_b = model.sign(key_b, signer_id="node-b")
        assert model.verify_signature(key_a.public_key(), sig_a["signature"]) is True
        assert model.verify_signature(key_b.public_key(), sig_b["signature"]) is True
        assert model.verify_signature(Ed25519PrivateKey.generate().public_key(), sig_a["signature"]) is False
