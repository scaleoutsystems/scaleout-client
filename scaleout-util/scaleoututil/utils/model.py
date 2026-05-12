import io
import json
import os
import shutil
import tempfile
import uuid
import warnings
import zipfile
from contextlib import contextmanager
from typing import BinaryIO, Iterable, Optional

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.utils.checksum import compute_checksum_from_stream
from scaleoututil.utils.signing import compute_model_signature, verify_model_signature
from scaleoututil.helpers.helpers import get_helper
from scaleoututil.helpers.plugins.numpyhelper import Helper

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB chunk size for reading/writing files

_ZIP_MAGIC = b"PK"
_FORMAT_VERSION = 2


_TRAINING_MODEL_ENTRY = "training_model.bin"
_METADATA_ENTRY = "metadata.json"
_INFERENCE_MODEL_ENTRY = "inference_model.bin"


def _read_metadata_from_zip(zip_path: str) -> dict:
    """Read and return metadata.json from a ZIP file, stripping the version key."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        if _METADATA_ENTRY in zf.namelist():
            meta = json.loads(zf.read(_METADATA_ENTRY).decode("utf-8"))
            meta.pop("version", None)
            return meta
    return {}


class ScaleoutModel:
    """The ScaleoutModel class is the primary model representation in the Scaleout framework.

    A ScaleoutModel is a self-describing, immutable container holding:
    - The serialized model weights (raw bytes, via a helper)
    - An optional full model representation (e.g. ONNX, .pt, …)
    - A metadata dict (eagerly loaded): model_id, helper, inference_model_format, and any user-supplied key-value pairs

    Models are always immutable after construction. To create a modified copy, use
    :meth:`to_builder` to obtain a pre-seeded :class:`ScaleoutModelBuilder`::

        new_model = model.to_builder().set_model_id("m-002").build()
        new_model = model.to_builder().set_metadata("tag", "v2").build()

    API
    ---
    Reading metadata::

        model.metadata          # shallow copy of the metadata dict
        model.model_id          # convenience property

    Model parameters::

        with model.get_training_model_stream() as s:  # read directly from ZIP, no temp file
            data = s.read()
        params = model.get_training_model(helper)     # decodes from ZIP

    Full model representation::

        model.has_inference_model                         # bool
        model.get_inference_model_format()                # format string or None
        with model.get_inference_model_stream() as s:    # stream inference_model.bin directly from ZIP
            data = s.read()

    On-disk format
    --------------
    Files are ZIP archives (stdlib zipfile, DEFLATED) containing:
        metadata.json    – eagerly loaded on open
        training_model.bin        – serialized weights
        inference_model.bin   – optional full model representation

    Legacy raw-binary streams (no ZIP header, e.g. bare NPZ) are detected automatically
    and loaded with empty metadata for backward compatibility.

    Storage
    -------
    A single ZIP file (``_zip_path``) is the canonical representation from construction
    time onwards.  The model always owns its ZIP and deletes it on garbage collection.

    ``get_training_model_stream()`` and ``get_inference_model_stream()`` stream entries directly
    from the ZIP without extracting to a temp file.
    """

    def __init__(self):
        raise TypeError(
            "ScaleoutModel cannot be instantiated directly. Use a factory method: from_training_model(), "
            "from_file(), from_stream(), or from_filechunk_stream() or use ScaleoutModelBuilder for more control."
        )

    @classmethod
    def _create(cls) -> "ScaleoutModel":
        """Internal factory — bypasses __init__ to construct a blank instance."""
        obj = object.__new__(cls)
        # ZIP file — always the canonical representation; always owned
        obj._zip_path = None
        obj._zip_is_owned = False
        # Open file handle held for the model's lifetime so the OS knows the file is in use
        obj._file_handle = None
        # Metadata dict – always in memory, eagerly available
        obj._metadata = {}
        obj._helper = None
        obj._checksum = None
        obj._legacy_source = False  # True if loaded from legacy raw binary (no ZIP)
        return obj

    def __del__(self):
        self._cleanup()

    def __repr__(self) -> str:
        mid = self._metadata.get("model_id", "?")
        helper = self._metadata.get("helper_type", "?")
        return f"<ScaleoutModel model_id={mid} helper={helper}>"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cleanup()

    def _cleanup(self):
        fh = getattr(self, "_file_handle", None)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
            self._file_handle = None
        zip_path = getattr(self, "_zip_path", None)
        if zip_path and getattr(self, "_zip_is_owned", False):
            try:
                os.unlink(zip_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> Optional[str]:
        return self._metadata.get("model_id")

    @property
    def legacy_source(self) -> bool:
        """True if this model was loaded from a legacy raw binary (no ZIP)."""
        return self._legacy_source

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Returns a shallow copy of the metadata dict."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------

    @property
    def has_training_model(self) -> bool:
        """True if the model contains serialized parameters."""
        if self._zip_path is not None:
            try:
                with zipfile.ZipFile(self._zip_path, "r") as zf:
                    return _TRAINING_MODEL_ENTRY in zf.namelist()
            except Exception:
                pass
        return False

    @contextmanager
    def get_training_model_stream(self):
        """Context manager yielding a read stream of the raw model binary.

        Reads directly from the ZIP entry — no temp-file extraction.

        Usage::

            with model.get_training_model_stream() as stream:
                data = stream.read()
        """
        with zipfile.ZipFile(self._zip_path, "r") as zf, zf.open(_TRAINING_MODEL_ENTRY) as f:
            yield f

    def get_training_model(self, helper=None):
        """Decodes and returns the model parameters via the helper."""
        self._helper = helper or self._helper
        if self._helper is None:
            raise ValueError("No helper provided to unpack model parameters.")
        with self.get_training_model_stream() as s:
            return self._helper.load(s)

    def get_model_params(self, helper=None):
        """Alias for get_training_model() for backward compatibility."""
        warnings.warn("`get_model_params` is deprecated. Use `get_training_model` instead.", DeprecationWarning, stacklevel=2)
        return self.get_training_model(helper)

    # ------------------------------------------------------------------
    # Full model representation
    # ------------------------------------------------------------------

    @property
    def has_inference_model(self) -> bool:
        """True if the model contains a full model representation."""
        if self._zip_path is not None:
            try:
                with zipfile.ZipFile(self._zip_path, "r") as zf:
                    return _INFERENCE_MODEL_ENTRY in zf.namelist()
            except Exception:
                pass
        return False

    def get_inference_model_format(self) -> Optional[str]:
        """Returns the format string of the full model representation, or ``None``."""
        return self._metadata.get("inference_model_format")

    @contextmanager
    def get_inference_model_stream(self):
        """Context manager yielding a read stream of the full model representation binary.

        Reads directly from the ZIP entry — no temp-file extraction.

        Usage::

            if model.has_inference_model:
                with model.get_inference_model_stream() as stream:
                    data = stream.read()
        """
        with zipfile.ZipFile(self._zip_path, "r") as zf, zf.open(_INFERENCE_MODEL_ENTRY) as f:
            yield f

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    @property
    def checksum(self) -> str:
        if self._checksum is None:
            with self.get_file_stream() as s:
                self._checksum = compute_checksum_from_stream(s)
        return self._checksum

    def verify_checksum(self, checksum: str) -> bool:
        return checksum is None or self.checksum == checksum

    # ------------------------------------------------------------------
    # Signature
    # ------------------------------------------------------------------

    def sign(self, private_key, signer_id: Optional[str] = None) -> dict:
        """Compute and return a signature dict for this model.

        Does not modify the ZIP.  POST the returned dict to
        ``/api/v1/model-signatures/`` to persist it.

        Args:
            private_key: An ``Ed25519PrivateKey`` from the ``cryptography`` package.
            signer_id: Optional free-form string identifying the signer.

        Returns:
            A dict with keys ``model_id``, ``algorithm``, ``signature``
            (base64-encoded), and optionally ``signer_id``.
        """
        sig = compute_model_signature(self._zip_path, private_key, signer_id)
        sig["model_id"] = self.model_id
        return sig

    def verify_signature(self, public_key, signature: str) -> bool:
        """Return ``True`` if *signature* is valid for this model.

        Args:
            public_key: An ``Ed25519PublicKey`` from the ``cryptography`` package.
            signature: Base64-encoded signature string — as returned by
                ``GET /api/v1/model-signatures/<id>`` ``signature`` field.
        """
        return verify_model_signature(self._zip_path, public_key, signature)

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def to_builder(self) -> "ScaleoutModelBuilder":
        """Return a factory pre-loaded from this model's current state.

        Use this to create a modified copy without mutating the original::

            new_model = model.to_builder().set_model_id("m-002").build()
            new_model = model.to_builder().set_metadata("tag", "v2").build()
        """
        # Copy to a temp file so the builder owns its own working copy and the
        # original model's file is not affected by the builder's lifecycle.
        f = ScaleoutModelBuilder()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.close()
        shutil.copy2(self._zip_path, tmp.name)
        f._loaded_zip_path = tmp.name
        f._loaded_zip_owned = True
        f._loaded_metadata = dict(self._metadata)
        f._legacy_source = self._legacy_source
        return f

    # ------------------------------------------------------------------
    # ZIP format detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(stream: io.BytesIO) -> str:
        """Returns 'zip' if data is our ZIP container format, else 'legacy'.

        Legacy NPZ files (produced by NumpyHelper) are also ZIP-based and start
        with the PK magic bytes.  We distinguish them from our container by
        checking for the presence of 'metadata.json' inside the archive.
        """
        data = stream.read(2)
        if data != _ZIP_MAGIC:
            stream.seek(0)
            return "legacy"
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream, "r") as zf:
                if _METADATA_ENTRY in zf.namelist():
                    stream.seek(0)
                    return "zip"
        except zipfile.BadZipFile:
            pass
        stream.seek(0)
        return "legacy"

    @staticmethod
    def detect_format_file(file_path: str) -> str:
        """Returns 'zip' if the file is our ZIP container format, else 'legacy'.

        Reads only the central directory — does not load the entire file.
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if _METADATA_ENTRY in zf.namelist():
                    return "zip"
        except (zipfile.BadZipFile, OSError):
            pass
        return "legacy"

    # ------------------------------------------------------------------
    # Factory methods (backward-compatible wrappers — delegate to ScaleoutModelBuilder)
    # ------------------------------------------------------------------

    @staticmethod
    def from_training_model(
        model_params,
        helper=None,
        metadata: Optional[dict] = None,
    ) -> "ScaleoutModel":
        """Creates a ScaleoutModel from model parameters.

        Delegates to :class:`ScaleoutModelBuilder`.  A fresh ``model_id`` is always
        generated; any ``model_id`` key present in ``metadata`` is ignored.

        Args:
            model_params: Parameters accepted by the helper's ``save()`` method.
            helper: Serialization helper. Falls back to NumpyHelper if not provided.
            metadata: Optional initial metadata dict.
        """
        f = ScaleoutModelBuilder.from_training_model(model_params, helper)
        if metadata:
            f.set_metadata_dict(metadata)
        return f.build()

    @staticmethod
    def from_file(file_path: str) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a file (ZIP or legacy raw binary).

        Delegates to :class:`ScaleoutModelBuilder`.
        """
        return ScaleoutModelBuilder.from_file(file_path).build()

    @staticmethod
    def from_stream(stream: BinaryIO) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a stream (ZIP or legacy raw binary).

        Delegates to :class:`ScaleoutModelBuilder`.
        """
        return ScaleoutModelBuilder.from_stream(stream).build()

    @staticmethod
    def from_filechunk_stream(
        filechunk_stream: Iterable[scaleout_msg.FileChunk],
    ) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a gRPC FileChunk iterator.

        Delegates to :class:`ScaleoutModelBuilder`.
        """
        return ScaleoutModelBuilder.from_filechunk_stream(filechunk_stream).build()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, file_path: str):
        """Saves the model to a file in ZIP format."""
        if os.path.abspath(self._zip_path) == os.path.abspath(file_path):
            return
        shutil.copy2(self._zip_path, file_path)
        # Close the handle to the old path before potentially unlinking it
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except OSError:
                pass
            self._file_handle = None
        # Drop old temp if we owned it, then reference the saved file
        if self._zip_is_owned:
            try:
                os.unlink(self._zip_path)
            except OSError:
                pass
        self._zip_path = file_path
        self._zip_is_owned = False
        self._file_handle = open(file_path, "rb")

    def get_file_stream(self):
        """Returns a new read handle to the packed ZIP file."""
        return open(self._zip_path, "rb")

    def get_filechunk_stream(self, chunk_size=CHUNK_SIZE):
        """Yields gRPC FileChunk messages of the raw model bytes."""
        with open(self._zip_path, "rb") as stream:
            while chunk := stream.read(chunk_size):
                yield scaleout_msg.FileChunk(data=chunk)


class ScaleoutModelBuilder:
    """Builder for :class:`ScaleoutModel` instances.

    Initialise with a classmethod, configure with ``set_*`` calls, then call
    :meth:`build` to produce the :class:`ScaleoutModel`.

    Examples::

        # New model from parameters
        model = (
            ScaleoutModelBuilder.from_training_model(params, helper)
            .set_metadata("session_id", "s1")
            .set_inference_model(onnx_bytes, "onnx")
            .build()
        )

        # Load existing model, optionally enrich metadata
        model = (
            ScaleoutModelBuilder.from_file("/path/model.zip")
            .set_metadata("tag", "v2")
            .build()
        )

        # Create a modified copy of an existing model
        new_model = model.to_builder().set_model_id("m-002").build()

    ``build()`` model-id behaviour
    --------------------------------
    *New-model path* (``from_training_model``): a fresh ``model_id`` UUID is always
    generated.  Any ``model_id`` passed via ``set_metadata`` / ``set_metadata_dict``
    is silently ignored.

    *Load path* (``from_file``, ``from_stream``, ``from_filechunk_stream``,
    ``ScaleoutModel.to_builder()``): the existing ``model_id`` is preserved **unless**
    ``set_inference_model`` or ``set_model_id`` was called before ``build()``.
    """

    def __init__(self):
        self._helper = None
        self._helper_type: Optional[str] = None
        self._create_new: bool = False
        # Load-path state
        self._loaded_zip_path: Optional[str] = None
        self._loaded_zip_owned: bool = False  # True if we created/own the loaded ZIP temp file
        self._loaded_training_model_stream = None  # binary stream: set for legacy loads; mutually exclusive with _training_model
        self._loaded_metadata: dict = {}
        # Configuration
        self._training_model = None  # raw Python params; mutually exclusive with _loaded_training_model_stream
        self._extra_metadata: dict = {}
        self._inference_model_stream = None  # SpooledTemporaryFile or None
        self._inference_model_stream_fmt: Optional[str] = None
        self._model_id_explicitly_set: bool = False
        # Legacy source tracking (for backward compatibility with non-ZIP raw binary inputs)
        self._legacy_source = False  # True if loaded from legacy raw binary (no ZIP)

    # ------------------------------------------------------------------
    # Initialisation classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_training_model(cls, model_params, helper=None) -> "ScaleoutModelBuilder":
        """Initialise the factory with raw model parameters (new-model path)."""
        f = cls()
        f._create_new = True
        if helper is not None:
            f._helper = helper
        f._training_model = model_params
        return f

    @classmethod
    def from_file(cls, file_path: str) -> "ScaleoutModelBuilder":
        """Initialise the factory by loading a ZIP or legacy file.

        The original file is referenced directly — no copy is made.  Use
        :meth:`ScaleoutModel.to_builder` when you need a mutable working copy
        that is independent of the source file's lifetime.
        """
        f = cls()
        if ScaleoutModel.detect_format_file(file_path) == "zip":
            f._loaded_zip_path = os.path.abspath(file_path)
            f._loaded_zip_owned = False  # external file — we must not delete it
            f._loaded_metadata = _read_metadata_from_zip(file_path)
        else:
            # Legacy: copy into a spooled temp file so the file handle can be closed immediately
            buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
            with open(file_path, "rb") as fh:
                shutil.copyfileobj(fh, buf)
            buf.seek(0)
            return cls.from_stream(buf)
        return f

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> "ScaleoutModelBuilder":
        """Initialise the factory from a binary stream (ZIP or legacy raw binary)."""
        f = cls()
        if ScaleoutModel.detect_format(stream) == "zip":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp.close()
            with open(tmp.name, "wb") as fh:
                shutil.copyfileobj(stream, fh)
            f._loaded_zip_path = tmp.name
            f._loaded_zip_owned = True
            f._loaded_metadata = _read_metadata_from_zip(tmp.name)
        else:
            # Legacy raw binary: defer wrapping until build()
            f._loaded_training_model_stream = stream
            f._loaded_metadata = {}
            f._legacy_source = True
        return f

    @classmethod
    def from_filechunk_stream(
        cls,
        filechunk_stream: Iterable[scaleout_msg.FileChunk],
    ) -> "ScaleoutModelBuilder":
        """Initialise the factory from a gRPC FileChunk iterator."""
        buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)  # 10 MB threshold for in-memory buffering
        for chunk in filechunk_stream:
            if chunk.data:
                buf.write(chunk.data)
        buf.seek(0)
        return cls.from_stream(buf)

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value) -> "ScaleoutModelBuilder":
        """Set a single metadata key. Returns ``self`` for chaining."""
        self._extra_metadata[key] = value
        return self

    def set_metadata_dict(self, metadata: dict) -> "ScaleoutModelBuilder":
        """Merge a dict of metadata. Returns ``self`` for chaining."""
        self._extra_metadata.update(metadata)
        return self

    def set_model_id(self, model_id: str) -> "ScaleoutModelBuilder":
        """Override the model_id. Returns ``self`` for chaining.

        Suppresses auto-generation of a new ``model_id`` that would otherwise occur
        when metadata, model parameters, or the full model representation change.
        """
        self._extra_metadata["model_id"] = model_id
        self._model_id_explicitly_set = True
        return self

    def set_helper_type(self, helper_type: str) -> "ScaleoutModelBuilder":
        """Set the helper by type name. Returns ``self`` for chaining."""
        self._helper_type = helper_type
        self._helper = None
        return self

    def set_helper(self, helper) -> "ScaleoutModelBuilder":
        """Set the helper instance. Returns ``self`` for chaining."""
        self._helper = helper
        self._helper_type = None
        return self

    def set_training_model(self, model_params, helper=None) -> "ScaleoutModelBuilder":
        """Override the model parameters. Returns ``self`` for chaining.

        Stores the raw parameters for serialization at :meth:`build` time.
        Clears any previously set ``_loaded_training_model_stream``.

        Args:
            model_params: Parameters accepted by the helper's ``save()`` method.
            helper: Optional helper instance. Overrides any previously configured helper.
        """
        if helper is not None:
            self._helper = helper
            self._helper_type = None
        self._training_model = model_params
        self._loaded_training_model_stream = None
        return self

    def set_inference_model(self, data: bytes, fmt: str) -> "ScaleoutModelBuilder":
        """Attach a full model representation from bytes. Returns ``self`` for chaining.

        Args:
            data: Raw bytes of the full model representation.
            fmt: Format identifier string (e.g. ``'onnx'``, ``'pt'``).
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes.")
        buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
        buf.write(data)
        buf.seek(0)
        self._inference_model_stream = buf
        self._inference_model_stream_fmt = fmt
        return self

    def set_inference_model_stream(self, stream: BinaryIO, fmt: str) -> "ScaleoutModelBuilder":
        """Attach a full model representation from a stream. Returns ``self`` for chaining.

        Args:
            stream: Readable binary stream of the full model representation.
            fmt: Format identifier string (e.g. ``'onnx'``, ``'pt'``).
        """
        buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
        shutil.copyfileobj(stream, buf)
        buf.seek(0)
        self._inference_model_stream = buf
        self._inference_model_stream_fmt = fmt
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ScaleoutModel:
        """Construct and return the :class:`ScaleoutModel`."""
        zip_path, zip_is_owned, metadata = self._prepare_zip()
        return self._load_model(zip_path, zip_is_owned, metadata)

    # ------------------------------------------------------------------
    # Build step 1: prepare the ZIP file
    # ------------------------------------------------------------------

    def _prepare_zip(self) -> tuple:
        """Step 1: create or select the ZIP file.

        Returns ``(zip_path, zip_is_owned, metadata)``.

        Rebuilds a new ZIP whenever model parameters, full model, or metadata have
        changed — and always assigns a fresh model_id in that case.  Only when the
        loaded ZIP is reused unchanged is the existing model_id preserved.
        """
        # Resolve helper
        if self._helper is None and self._helper_type:
            self._helper = get_helper(self._helper_type)
        if self._helper is None:
            helper_type = self._loaded_metadata.get("helper_type")
            if helper_type:
                self._helper = get_helper(helper_type)

        # Serialize raw model params if provided (from_training_model / set_training_model path)
        training_model_stream = self._loaded_training_model_stream
        if self._training_model is not None:
            helper = self._helper if self._helper is not None else Helper()
            self._helper = helper
            buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
            helper.save(self._training_model, buf)
            buf.seek(0)
            training_model_stream = buf

        # Build metadata: start from loaded state, apply user overrides (model_id handled below)
        metadata: dict = dict(self._loaded_metadata)
        for k, v in self._extra_metadata.items():
            if k != "model_id":
                metadata[k] = v
        if self._helper:
            metadata.setdefault("helper_type", self._helper.name)
        if self._inference_model_stream is not None:
            metadata["inference_model_format"] = self._inference_model_stream_fmt

        needs_rebuild = training_model_stream is not None or self._inference_model_stream is not None or bool(self._extra_metadata)

        # Assign model_id: auto-generate when rebuilding; keep existing only when reusing the file as-is.
        # set_model_id() overrides auto-generation on the load path; ignored for from_training_model.
        if needs_rebuild:
            if self._model_id_explicitly_set and not self._create_new:
                metadata["model_id"] = self._extra_metadata["model_id"]
            else:
                metadata["model_id"] = str(uuid.uuid4())

        if not needs_rebuild:
            return self._loaded_zip_path, self._loaded_zip_owned, metadata

        zip_path = ScaleoutModelBuilder._build_zip(
            metadata,
            training_model_stream=training_model_stream,
            source_zip_path=self._loaded_zip_path,
            inference_model_stream=self._inference_model_stream,
        )
        if self._loaded_zip_owned and self._loaded_zip_path:
            try:
                os.unlink(self._loaded_zip_path)
            except OSError:
                pass
        return zip_path, True, metadata

    @staticmethod
    def _build_zip(
        metadata: dict,
        training_model_stream: Optional[BinaryIO] = None,
        source_zip_path: Optional[str] = None,
        inference_model_stream: Optional[BinaryIO] = None,
    ) -> str:
        """Build a new owned temp ZIP from components. Returns the temp file path."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            meta = dict(metadata)
            meta["version"] = _FORMAT_VERSION
            zf.writestr(_METADATA_ENTRY, json.dumps(meta))

            if training_model_stream is not None:
                with zf.open(_TRAINING_MODEL_ENTRY, "w", force_zip64=True) as dst_f:
                    shutil.copyfileobj(training_model_stream, dst_f)
            elif source_zip_path and os.path.exists(source_zip_path):
                with zipfile.ZipFile(source_zip_path, "r") as src_zf:
                    if _TRAINING_MODEL_ENTRY in src_zf.namelist():
                        with src_zf.open(_TRAINING_MODEL_ENTRY) as src_f, zf.open(_TRAINING_MODEL_ENTRY, "w", force_zip64=True) as dst_f:
                            shutil.copyfileobj(src_f, dst_f)

            if inference_model_stream is not None:
                with zf.open(_INFERENCE_MODEL_ENTRY, "w", force_zip64=True) as dst_f:
                    shutil.copyfileobj(inference_model_stream, dst_f)
            elif source_zip_path and os.path.exists(source_zip_path):
                with zipfile.ZipFile(source_zip_path, "r") as src_zf:
                    if _INFERENCE_MODEL_ENTRY in src_zf.namelist():
                        with src_zf.open(_INFERENCE_MODEL_ENTRY) as src_f, zf.open(_INFERENCE_MODEL_ENTRY, "w", force_zip64=True) as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Build step 2: load the ZIP into a ScaleoutModel
    # ------------------------------------------------------------------

    def _load_model(self, zip_path: str, zip_is_owned: bool, metadata: dict) -> ScaleoutModel:
        """Step 2: open the ZIP and populate a :class:`ScaleoutModel` instance."""
        model = ScaleoutModel._create()
        model._zip_path = zip_path
        model._zip_is_owned = zip_is_owned
        model._file_handle = open(zip_path, "rb")
        model._legacy_source = self._legacy_source
        model._metadata = metadata
        model._helper = self._helper  # already resolved in _prepare_zip
        return model
