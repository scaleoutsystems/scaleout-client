"""EdgeClient: user-facing interface for compute packages running on Scaleout Edge.

EdgeClient exposes the surface that user-written ``startup.py`` callbacks
should interact with: callback registration, metric/attribute/telemetry
logging, task-abort checks, and inference helpers. It owns an
:class:`EdgeClientRuntime` (a :class:`typing.Protocol`) that it delegates
connection, dispatch, and transport work to. The default runtime is
:class:`GrpcEdgeClientRuntime`; tests and alternative transports can inject
any object conforming to the protocol.
"""

import enum
import functools
import os
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import psutil

from scaleoututil.config import get_model_cache_dir

from scaleoututil.utils.dist import get_version as _get_version

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.queue import LinkQuality, PriorityClass
from scaleoututil.utils.model import ScaleoutModel

from scaleout.client.grpc_handler import GrpcConnectionOptions  # re-exported for back-compat
from scaleout.client.local_repository import LocalModelRepository
from scaleout.client.inference_context import InferenceContext
from scaleout.client.logging_context import LoggingContext

VERSION = _get_version("scaleout")

__all__ = [
    "EdgeClient",
    "EdgeClientRuntime",
    "ConnectToApiResult",
    "GracefulExitException",
    "GrpcConnectionOptions",
]


class ConnectToApiResult(enum.Enum):
    """Enum for representing the result of connecting to the Scaleout API."""

    Assigned = 0
    ComputePackageMissing = 1
    UnAuthorized = 2
    UnMatchedConfig = 3
    IncorrectUrl = 4
    UnknownError = 5


class GracefulExitException(Exception):
    pass


class EdgeClientRuntime(Protocol):
    """Structural contract for the runtime plugged into :class:`EdgeClient`.

    Any object matching this shape can be injected as the runtime. The
    production implementation is :class:`GrpcEdgeClientRuntime`; tests can
    supply mocks or recorders without inheriting from it.
    """

    @property
    def link_quality(self) -> LinkQuality: ...

    def send_metric(self, metrics: dict, model_id: str, step: int, round_id: str, session_id: str) -> bool: ...
    def send_attributes(self, attributes: dict, priority: Optional[str] = None) -> bool: ...
    def send_telemetry(
        self,
        key: Optional[str] = None,
        payload: Optional[dict] = None,
        telemetry: Optional[dict] = None,
        priority: Optional[str] = None,
    ) -> bool: ...
    def send_inference_result(self, key: str, payload: dict, inference_id: str, model_id: str, priority: Optional[str] = None) -> bool: ...
    def check_task_abort(self) -> None: ...
    def get_model_from_combiner(self, model_id: str) -> ScaleoutModel: ...
    def connect_to_api(
        self,
        url: str,
        json: Optional[dict] = None,
        token: Optional[str] = None,
        token_refresh_callback: Optional[Callable[..., None]] = None,
    ) -> Tuple["ConnectToApiResult", Any]: ...
    def init_grpchandler(
        self,
        config: GrpcConnectionOptions,
        token: Optional[str] = None,
        url: Optional[str] = None,
        token_refresh_callback: Optional[Callable[..., None]] = None,
    ) -> bool: ...
    def run(self, with_heartbeat: bool = True, with_polling: bool = True) -> None: ...
    def get_access_token(self) -> Optional[str]: ...
    def set_client(self, client: "EdgeClient") -> None: ...
    def list_priority_classes(self) -> list[PriorityClass]: ...
    def add_priority_class(self, cls: PriorityClass) -> None: ...
    def remove_priority_class(self, name: str, force: bool = False) -> int: ...
    def replace_priority_class(self, cls: PriorityClass) -> None: ...


class EdgeClient:
    """User-facing interface for an edge client.

    Users instantiate this class directly. The runtime defaults to
    :class:`GrpcEdgeClientRuntime`; passing ``runtime=`` at construction time
    swaps the implementation — useful for tests and alternative transports.
    """

    def __init__(
        self,
        train_callback: Optional[Callable[[ScaleoutModel, Dict], Tuple[Optional[ScaleoutModel], Dict]]] = None,
        validate_callback: Optional[Callable[[ScaleoutModel], Dict]] = None,
        runtime: Optional[EdgeClientRuntime] = None,
    ) -> None:
        """Initialize the EdgeClient."""
        self.name: Optional[str] = None
        self.client_id: Optional[str] = None
        self.package_path: str = "."

        # Optional cross-process compute lock. When ``SCALEOUT_CLIENT_COMPUTE_LOCK``
        # is set in the environment, train/validate callbacks are transparently
        # wrapped so only one client at a time runs the user's compute. This is
        # set up *before* assigning callbacks so wrapping applies to all entry
        # points (constructor args + set_*_callback).
        self._compute_lock = self._build_compute_lock()

        self.train_callback = self._maybe_wrap_with_lock(train_callback)
        self.validate_callback = self._maybe_wrap_with_lock(validate_callback)

        self.inference_callback: Optional[Callable[[ScaleoutModel, Dict], Any]] = None
        self.stage_model_callback: Optional[Callable[[ScaleoutModel], None]] = None

        self.registered_callbacks: Dict[str, Callable[[scaleout_msg.TaskRequest], Dict]] = {}

        self.local_repository = LocalModelRepository(repository_path=get_model_cache_dir())
        ScaleoutLogger().info(f"Scaleout version {VERSION}")

        self._current_logging_context = threading.local()
        self._current_inference_context = threading.local()
        self._telemetry_stop = threading.Event()

        if runtime is None:
            # Lazy import to break the edge_client <-> grpc_edge_client_runtime cycle.
            from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime  # noqa: PLC0415

            runtime = GrpcEdgeClientRuntime()

        self._runtime: EdgeClientRuntime = runtime
        runtime.set_client(self)

    # -- logging context -------------------------------------------------------

    @property
    def current_logging_context(self) -> Optional[LoggingContext]:
        """Get the current logging context for the running thread."""
        return getattr(self._current_logging_context, "value", None)

    @current_logging_context.setter
    def current_logging_context(self, context: LoggingContext) -> None:
        """Set the current logging context for the running thread."""
        self._current_logging_context.value = context

    @contextmanager
    def logging_context(self, context: LoggingContext):
        """Set the logging context for the duration of the block."""
        prev_context = self.current_logging_context
        self.current_logging_context = context
        try:
            yield
        finally:
            self.current_logging_context = prev_context

    # -- inference context -----------------------------------------------------

    @property
    def current_inference_context(self) -> Optional[InferenceContext]:
        """Get the current inference context for the running thread."""
        return getattr(self._current_inference_context, "value", None)

    @current_inference_context.setter
    def current_inference_context(self, context: InferenceContext) -> None:
        """Set the current inference context for the running thread."""
        self._current_inference_context.value = context

    @contextmanager
    def inference_context(self, context: InferenceContext):
        """Set the inference context for the duration of the block."""
        prev_context = self.current_inference_context
        self.current_inference_context = context
        try:
            yield
        finally:
            self.current_inference_context = prev_context

    # -- identity --------------------------------------------------------------

    def set_name(self, name: str) -> None:
        """Set the client name."""
        ScaleoutLogger().info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        ScaleoutLogger().info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    # -- link quality ----------------------------------------------------------

    @property
    def link_quality(self) -> LinkQuality:
        """Current client-internal link quality derived from the runtime."""
        return self._runtime.link_quality

    # -- default telemetry loop ------------------------------------------------

    _TELEMETRY_HEALTHY_INTERVAL = 5.0
    _TELEMETRY_DEGRADED_INTERVAL = 30.0

    def _default_telemetry_loop(self) -> None:
        """Emit memory and CPU telemetry until stop_default_telemetry_loop is called.

        Stretches the sampling interval to 30 s when the link is DEGRADED so
        stale system-health samples do not crowd out higher-priority traffic.
        """
        self._telemetry_stop.clear()
        while not self._telemetry_stop.is_set():
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            try:
                self.log_telemetry(key="memory_usage", payload={"value": memory_usage})
                self.log_telemetry(key="cpu_usage", payload={"value": cpu_usage})
            except Exception as e:
                ScaleoutLogger().warning(f"Enqueueing telemetry failed: {e}")
            interval = (
                self._TELEMETRY_DEGRADED_INTERVAL
                if self.link_quality == LinkQuality.DEGRADED or self.link_quality == LinkQuality.OFFLINE
                else self._TELEMETRY_HEALTHY_INTERVAL
            )
            if self._telemetry_stop.wait(interval):
                break

    def stop_default_telemetry_loop(self) -> None:
        """Signal default_telemetry_loop to exit at its next sleep boundary."""
        self._telemetry_stop.set()

    # -- callback registration -------------------------------------------------

    def set_train_callback(self, callback: callable) -> None:
        """Register the callback invoked when a training task request arrives.

        Called by the client with the current global model each time the
        combiner dispatches a training request. The callback should perform
        the local training update and return the new model.

        Args:
            callback (callable): ``(scaleout_model, settings) -> (model, metadata)``

                - ``scaleout_model`` (ScaleoutModel): The current model to train.
                  Load parameters with ``scaleout_model.get_training_model(helper)``.
                - ``settings`` (dict): Training settings for this round (e.g.
                  epochs, batch size, learning rate).
                - Returns a tuple of the updated model and a metadata dict. The
                  metadata dict is used by the aggregator and for logging; for
                  the default aggregators (fedavg, fedopt) it must at least
                  contain ``{"training_metadata": {"num_examples": int}}``.

                Call ``self.check_task_abort()`` periodically during training
                (e.g. once few iterations) to allow the task to be stopped
                gracefully if the session is terminated from the server, and
                use ``self.log_metric(...)`` to report progress in real time.
        """
        self.train_callback = self._maybe_wrap_with_lock(callback)

    def set_validate_callback(self, callback: callable) -> None:
        """Register the callback invoked when a validation task request arrives.

        Called by the client after a new global model has been produced, so
        the callback can validate that model and report metrics. Registering
        this callback is optional.

        Args:
            callback (callable): ``(scaleout_model) -> metrics``

                - ``scaleout_model`` (ScaleoutModel): The model to validate.
                  Load parameters with ``scaleout_model.get_training_model(helper)``.
                - Returns a dict of validation metrics. Scalar entries are
                  captured and visualized in the Scaleout Edge UI; the entire
                  dict is stored in the backend and accessible via the API/UI.
        """
        self.validate_callback = self._maybe_wrap_with_lock(callback)

    # -- compute lock ----------------------------------------------------------
    # When SCALEOUT_CLIENT_COMPUTE_LOCK is set in the environment, train and
    # validate callbacks are wrapped in ``with FileLock(path):`` so siblings
    # that share the same lock path execute compute one at a time. The
    # external orchestrator (e.g. scaleoututil.launchers.launch_clients) sets
    # this env var; user-written client packages don't need to know about it.
    # This bounds peak memory when N clients share one CPU/GPU.

    @staticmethod
    def _build_compute_lock():
        path = os.environ.get("SCALEOUT_CLIENT_COMPUTE_LOCK")
        if not path:
            return None
        try:
            from filelock import FileLock  # noqa: PLC0415 - optional dep
        except ImportError as e:
            raise RuntimeError("SCALEOUT_CLIENT_COMPUTE_LOCK is set but the 'filelock' package is not installed in this environment.") from e
        ScaleoutLogger().info(f"Serializing train/validate compute via {path}")
        return FileLock(path)

    def _maybe_wrap_with_lock(self, callback):
        if callback is None or self._compute_lock is None:
            return callback
        lock = self._compute_lock

        @functools.wraps(callback)
        def wrapped(*args, **kwargs):
            with lock:
                return callback(*args, **kwargs)

        return wrapped

    def set_inference_callback(self, callback: Callable[[ScaleoutModel, Dict], Any]) -> None:
        """Set the inference callback."""
        self.inference_callback = callback

    def set_stage_model_callback(self, callback: Callable[[ScaleoutModel], None]) -> None:
        """Set the stage-model callback, invoked after a model is staged for inference."""
        self.stage_model_callback = callback

    def set_custom_callback(self, callback_name: str, callback: Callable[[scaleout_msg.TaskRequest], Dict]) -> None:
        """Set a custom task callback."""
        if not callback_name.startswith("Custom_"):
            callback_name = "Custom_" + callback_name
        self.registered_callbacks[callback_name] = callback
        ScaleoutLogger().info(f"Registered custom callback: {callback_name}")

    def remove_custom_callback(self, callback_name: str) -> None:
        """Remove a custom task callback."""
        if not callback_name.startswith("Custom_"):
            callback_name = "Custom_" + callback_name
        if callback_name in self.registered_callbacks:
            del self.registered_callbacks[callback_name]
            ScaleoutLogger().info(f"Removed custom callback: {callback_name}")
        else:
            ScaleoutLogger().warning(f"Custom callback {callback_name} not found")

    # -- reporting -------------------------------------------------------------

    def log_metric(self, metrics: dict, step: int = None, commit: bool = True, check_task_abort: bool = True, context: LoggingContext = None) -> bool:
        """Log the metrics to the server.

        Args:
            metrics (dict): The metrics to log.
            step (int, optional): The step number.
            If provided the context step will be set to this value.
            If not provided, the step from the context will be used.
            commit (bool, optional): Whether or not to increment the step.  Defaults to True.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.
            context (LoggingContext, optional): The logging context to use. Defaults to None, which uses the current context.

        Returns:
            bool: True if the metrics were logged successfully, False otherwise.

        """
        context = context or self.current_logging_context

        if context is None:
            ScaleoutLogger().error("Missing context for logging metric.")
            return False

        if step is None:
            step = context.step
        else:
            context.step = step

        if commit:
            context.step += 1

        success = self._runtime.send_metric(
            metrics=metrics,
            model_id=context.model_id,
            step=step,
            round_id=context.round_id,
            session_id=context.session_id,
        )
        if check_task_abort:
            self._runtime.check_task_abort()
        return success

    def log_attributes(self, attributes: dict, priority: Optional[str] = None, check_task_abort: bool = True) -> bool:
        """Log the attributes to the server.

        Args:
            attributes (dict): The attributes to log.
            priority (str, optional): Priority class name (e.g. "artifact"). Defaults to "artifact".
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the attributes were logged successfully, False otherwise.

        """
        success = self._runtime.send_attributes(attributes, priority=priority)
        if check_task_abort:
            self._runtime.check_task_abort()
        return success

    def log_telemetry(
        self,
        key: Optional[str] = None,
        payload: Optional[dict] = None,
        telemetry: Optional[dict] = None,
        priority: Optional[str] = None,
        check_task_abort: bool = True,
    ) -> bool:
        """Log a telemetry observation to the server.

        Intended usage: ``log_telemetry(key="loss", payload={"value": 0.5, "step": 12})``.

        Args:
            key (str): Telemetry key (e.g. metric name).
            payload (dict, optional): JSON payload — typically ``{"value": <float>, ...}``
                for plottable scalars, or arbitrary structured data for coupled
                observations (coordinates, bundles).
            telemetry (dict, optional): **Legacy** — mapping of key to scalar value.
                In record mode this emits a warning and expands to one record per pair.
                In legacy mode (``SCALEOUT_CLIENT_LEGACY_TELEMETRY=true``) it is the
                accepted shape.
            priority (str, optional): Priority class name (e.g. "alert"). Defaults to "telemetry".
            check_task_abort (bool, optional): Whether to check for task abort. Defaults to True.

        Returns:
            bool: True if the telemetry was accepted into the queue.

        Backward compat: existing positional callers passing a dict
        (``log_telemetry({"loss": 0.5})``) are accepted; the dict is routed
        through the legacy ``telemetry`` slot.
        """
        # Back-compat: existing positional callers pass a dict where `key` now lives.
        if isinstance(key, dict):
            warnings.warn(
                "log_telemetry: passing a telemetry dict positionally is deprecated; use 'key' (+ 'payload') instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if telemetry is not None:
                ScaleoutLogger().warning("log_telemetry: both positional dict and 'telemetry=' supplied; using positional")
            telemetry = key
            key = None
        success = self._runtime.send_telemetry(key=key, payload=payload, telemetry=telemetry, priority=priority)
        if check_task_abort:
            self._runtime.check_task_abort()
        return success

    def log_inference_result(
        self,
        key: str,
        payload: dict,
        context: Optional[InferenceContext] = None,
        priority: Optional[str] = None,
    ) -> bool:
        """Log an inference result (sighting/observation) to the server.

        Args:
            key (str): A label for this result type (e.g. "detection", "classification").
            payload (dict): JSON-serialisable observation data.
            context (InferenceContext, optional): Override the thread-local inference context.
            priority (str, optional): Priority class name (e.g. "telemetry"). Defaults to "inference".

        Returns:
            bool: True if the result was accepted into the queue, False if no context is set.
        """
        context = context or self.current_inference_context
        if context is None:
            ScaleoutLogger().warning("log_inference_result called without an active InferenceContext; result dropped")
            return False
        return self._runtime.send_inference_result(
            key=key,
            payload=payload,
            inference_id=context.inference_id,
            model_id=context.model_id,
            priority=priority,
        )

    def check_task_abort(self) -> None:
        """Check if the ongoing task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from a thread that do not run the task, this function is a no-op.

        Raises:
            StoppedException: If the task was aborted.

        """
        self._runtime.check_task_abort()

    # -- priority class management --------------------------------------------

    @property
    def priority_classes(self) -> list[PriorityClass]:
        """Return the currently registered priority classes (excludes internal default)."""
        return self._runtime.list_priority_classes()

    def add_priority_class(self, cls: PriorityClass) -> None:
        """Register a new priority class for use with log_* methods.

        Args:
            cls (PriorityClass): The class to register. Its name must be unique.

        Raises:
            ValueError: If a class with that name already exists.
        """
        self._runtime.add_priority_class(cls)

    def remove_priority_class(self, name: str, force: bool = False) -> int:
        """Remove a registered priority class.

        Args:
            name (str): The class name to remove.
            force (bool): If True, migrate queued records to the default class.
                If False, raises ValueError when the class is non-empty.

        Returns:
            int: Number of records migrated to the default class.
        """
        return self._runtime.remove_priority_class(name, force)

    def replace_priority_class(self, cls: PriorityClass) -> None:
        """Update an existing priority class definition in-place.

        Queued records are not moved — they pick up the new parameters on the
        next scheduling pass.

        Args:
            cls (PriorityClass): Replacement definition. Its name must match an existing class.

        Raises:
            KeyError: If no class with that name exists.
        """
        self._runtime.replace_priority_class(cls)

    # -- connection / lifecycle (delegated) -----------------------------------

    def connect_to_api(
        self,
        url: str,
        json: Optional[dict] = None,
        token: Optional[str] = None,
        token_refresh_callback: Optional[Callable[..., None]] = None,
    ) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the Scaleout API via the runtime."""
        return self._runtime.connect_to_api(url=url, json=json, token=token, token_refresh_callback=token_refresh_callback)

    def init_grpchandler(
        self,
        config: GrpcConnectionOptions,
        token: Optional[str] = None,
        url: Optional[str] = None,
        token_refresh_callback: Optional[Callable[..., None]] = None,
    ) -> bool:
        """Initialize the runtime's transport handler."""
        return self._runtime.init_grpchandler(config=config, token=token, url=url, token_refresh_callback=token_refresh_callback)

    def run(self, with_heartbeat: bool = True, with_polling: bool = True) -> None:
        """Run the client's event loop via the runtime."""
        self._runtime.run(with_heartbeat=with_heartbeat, with_polling=with_polling)

    def get_access_token(self) -> Optional[str]:
        """Return the current access token, if the runtime manages one."""
        return self._runtime.get_access_token()

    # -- inference -------------------------------------------------------------

    def stage_model(self, model: ScaleoutModel | str) -> ScaleoutModel:
        """Stage a model for inference.

        :param model: The ScaleoutModel or model id to stage.
        """
        if self.local_repository.model_exists(model):
            ScaleoutLogger().info(f"Model {model} already staged in local repository.")
        else:
            if isinstance(model, str):
                downloaded_model = self._runtime.get_model_from_combiner(model_id=model)
                if downloaded_model is None:
                    raise ValueError(f"Model with ID {model} not found in combiner.")
                model = downloaded_model
            self.local_repository.stage_model(model)
        if isinstance(model, str):
            model = self.local_repository.get_model_by_id(model)
        return model

    def run_inference(self, model: ScaleoutModel | str = None, params: Dict = None) -> None:
        """Run inference using the specified model.

        :param model: The ScaleoutModel or model ID string to use for inference.
        :param params: Additional parameters for inference.
        """
        if self.inference_callback is None:
            raise ValueError("No inference callback set")

        if isinstance(model, str):
            model = self.stage_model(model)

        if model is None:
            raise ValueError("Model not found in repository.")
        return self.inference_callback(model, params)
