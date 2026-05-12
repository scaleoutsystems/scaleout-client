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
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

from scaleoututil.utils.dist import get_version as _get_version

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.model import ScaleoutModel

from scaleout.client.grpc_handler import GrpcConnectionOptions  # re-exported for back-compat
from scaleout.client.local_repository import LocalModelRepository
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

    def send_metric(self, metrics: dict, model_id: str, step: int, round_id: str, session_id: str) -> bool: ...
    def send_attributes(self, attributes: dict) -> bool: ...
    def send_telemetry(self, telemetry: dict) -> bool: ...
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
    def run(self, with_heartbeat: bool = False, with_polling: bool = True) -> None: ...
    def get_access_token(self) -> Optional[str]: ...


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

        self.train_callback = train_callback
        self.validate_callback = validate_callback

        self.inference_callback: Optional[Callable[[ScaleoutModel, Dict], Any]] = None
        self.stage_model_callback: Optional[Callable[[ScaleoutModel], None]] = None

        self.registered_callbacks: Dict[str, Callable[[scaleout_msg.TaskRequest], Dict]] = {}

        self.local_repository = LocalModelRepository(repository_path="./.model_cache")
        ScaleoutLogger().info(f"Scaleout version {VERSION}")

        self._current_logging_context = threading.local()

        if runtime is None:
            # Lazy import to break the edge_client <-> grpc_edge_client_runtime cycle.
            from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime  # noqa: PLC0415

            runtime = GrpcEdgeClientRuntime(self)
        self._runtime: EdgeClientRuntime = runtime

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

    # -- identity --------------------------------------------------------------

    def set_name(self, name: str) -> None:
        """Set the client name."""
        ScaleoutLogger().info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        ScaleoutLogger().info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    # -- callback registration -------------------------------------------------

    def set_train_callback(self, callback: callable) -> None:
        """Set the train callback."""
        self.train_callback = callback

    def set_validate_callback(self, callback: callable) -> None:
        """Set the validate callback."""
        self.validate_callback = callback

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

    def log_attributes(self, attributes: dict, check_task_abort: bool = True) -> bool:
        """Log the attributes to the server.

        Args:
            attributes (dict): The attributes to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the attributes were logged successfully, False otherwise.

        """
        success = self._runtime.send_attributes(attributes)
        if check_task_abort:
            self._runtime.check_task_abort()
        return success

    def log_telemetry(self, telemetry: dict, check_task_abort: bool = True) -> bool:
        """Log the telemetry data to the server.

        Args:
            telemetry (dict): The telemetry data to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the telemetry data was logged successfully, False otherwise.

        """
        success = self._runtime.send_telemetry(telemetry)
        if check_task_abort:
            self._runtime.check_task_abort()
        return success

    def check_task_abort(self) -> None:
        """Check if the ongoing task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from a thread that do not run the task, this function is a no-op.

        Raises:
            StoppedException: If the task was aborted.

        """
        self._runtime.check_task_abort()

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

    def run(self, with_heartbeat: bool = False, with_polling: bool = True) -> None:
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
