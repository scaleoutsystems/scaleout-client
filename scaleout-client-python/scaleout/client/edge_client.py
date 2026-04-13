"""EdgeClient class for interacting with the Scaleout network."""

import enum
import json
import signal
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime

from scaleout.utils.dist import VERSION
from scaleoututil.utils.model import ScaleoutModel
from scaleoututil.utils.url import assemble_endpoint_url
import psutil
import requests

from scaleoututil.auth.token_manager import TokenManager

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.config import (
    SCALEOUT_AUTH_SCHEME,
    SCALEOUT_CONNECT_API_SECURE,
    SCALEOUT_CLIENT_STATUS_REPORTING,
    SCALEOUT_CLIENT_SEND_TELEMETRY,
    SCALEOUT_GRACEFUL_CLIENT_CONNECTION,
    SCALEOUT_CHECK_COMPATIBILITY,
    SCALEOUT_CLIENT_TASK_POLLING_INTERVAL,
)
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.grpc_handler import GrpcConnectionOptions, GrpcHandler, RetryException
from scaleoututil.utils.http_status_codes import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_ACCEPTABLE,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_PACKAGE_MISSING,
    HTTP_STATUS_UNAUTHORIZED,
    HTTP_STATUS_SERVER_ERROR,
)
from scaleout.client.logging_context import LoggingContext
from scaleout.client.task_receiver import StoppedException, TaskReceiver, UnknownTaskType
from scaleoututil.grpc.tasktype import TaskType

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


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


class EdgeClient:
    """Client for interacting with the Scaleout network."""

    def __init__(
        self,
        train_callback: Optional[Callable[[ScaleoutModel, Dict], Tuple[Optional[ScaleoutModel], Dict]]] = None,
        validate_callback: Optional[Callable[[ScaleoutModel], Dict]] = None,
        predict_callback: Optional[Callable[[ScaleoutModel], Dict]] = None,
    ) -> None:
        """Initialize the EdgeClient."""
        self.name: str = None
        self.client_id: str = None

        self.train_callback = train_callback
        self.validate_callback = validate_callback
        self.predict_callback = predict_callback

        self.grpc_handler: Optional[GrpcHandler] = None
        self.package_path: str = "."

        self._current_logging_context = threading.local()
        self.task_receiver = TaskReceiver(self, self._run_task_callback, polling_interval=SCALEOUT_CLIENT_TASK_POLLING_INTERVAL)
        self.registered_callbacks: Dict[str, Callable[[scaleout_msg.TaskRequest], Dict]] = {}
        self.token_manager: Optional[TokenManager] = None

    @property
    def current_logging_context(self) -> Optional[LoggingContext]:
        """Get the current logging context for the running thread."""
        return getattr(self._current_logging_context, "value", None)

    @current_logging_context.setter
    def current_logging_context(self, context: LoggingContext) -> None:
        """Set the current logging context for the running thread."""
        self._current_logging_context.value = context

    def set_train_callback(self, callback: callable) -> None:
        """Set the train callback."""
        self.train_callback = callback

    def set_validate_callback(self, callback: callable) -> None:
        """Set the validate callback."""
        self.validate_callback = callback

    def set_predict_callback(self, callback: callable) -> None:
        """Set the predict callback."""
        self.predict_callback = callback

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

    def _get_current_token(self) -> Optional[str]:
        """Get the current access token, refreshing if needed."""
        if self.token_manager:
            return self.token_manager.get_access_token()
        return None

    def _init_token_manager(self, token: str, url: str, token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None) -> None:
        """Initialize the token manager with the provided token."""
        if self.token_manager is None:
            token_endpoint = assemble_endpoint_url(url, "api/auth", "refresh")
            self.token_manager = TokenManager(refresh_token=token, token_endpoint=token_endpoint, on_token_refresh=token_refresh_callback)

    def connect_to_api(
        self, url: str, json: dict = None, token: Optional[str] = None, token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None
    ) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the Scaleout API. Accepts a refresh token, instantiates TokenManager, and uses access token."""
        if token:
            self._init_token_manager(token, url, token_refresh_callback)
        current_token = self._get_current_token()

        url_endpoint = assemble_endpoint_url(url, "api/v1/clients/add")
        ScaleoutLogger().info(f"Connecting to API endpoint: {url_endpoint}")

        if SCALEOUT_CHECK_COMPATIBILITY:
            json["client_version"] = VERSION

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {current_token}"},
                timeout=REQUEST_TIMEOUT,
                verify=SCALEOUT_CONNECT_API_SECURE,
            )

            if response.status_code == HTTP_STATUS_OK:
                ScaleoutLogger().info("Connect to Scaleout API - Client assigned to controller")
                json_response = response.json()
                self.set_client_id(json_response["client_id"])
                self.set_name(json.get("name", json_response["client_id"]))
                combiner_config = GrpcConnectionOptions.from_dict(json_response)
                return ConnectToApiResult.Assigned, combiner_config

            if response.status_code == HTTP_STATUS_PACKAGE_MISSING:
                json_response = response.json()
                ScaleoutLogger().info("Connect to Scaleout API - Remote compute package missing.")
                return ConnectToApiResult.ComputePackageMissing, json_response

            if response.status_code == HTTP_STATUS_UNAUTHORIZED:
                ScaleoutLogger().error("Connect to Scaleout API - Unauthorized")
                return ConnectToApiResult.UnAuthorized, "Unauthorized"

            if response.status_code == HTTP_STATUS_BAD_REQUEST:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
                msg = json_response.get("message", "Unknown error")
                ScaleoutLogger().error(f"Connect to Scaleout API - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if response.status_code == HTTP_STATUS_NOT_ACCEPTABLE:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
                msg = json_response.get("message", "Unknown error")
                ScaleoutLogger().error(f"Connect to Scaleout API - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if response.status_code == HTTP_STATUS_NOT_FOUND:
                ScaleoutLogger().error("Connect to Scaleout API - Incorrect URL")
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"

            if response.status_code == HTTP_STATUS_SERVER_ERROR:
                response_json = response.json()
                msg = response_json.get("message", "Unknown server error")
                ScaleoutLogger().error(f"Connect to Scaleout API - Server error: {msg}")
                return ConnectToApiResult.UnknownError, f"Server error: {msg}"

        except Exception as e:
            ScaleoutLogger().error(f"Connect to Scaleout API - Error occurred: {str(e)}")
            return ConnectToApiResult.UnknownError, str(e)

    def init_grpchandler(
        self,
        config: GrpcConnectionOptions,
        token: Optional[str] = None,
        url: Optional[str] = None,
        token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> bool:
        """Initialize the GRPC handler. Accepts a refresh token, instantiates TokenManager, and uses access token."""
        if token and url:
            self._init_token_manager(token, url, token_refresh_callback)
        try:
            self.grpc_handler = GrpcHandler(self, host=config.host, port=config.port)

            if SCALEOUT_CHECK_COMPATIBILITY:
                success, server_version, msg = self.grpc_handler.check_version_compatibility()
                if not success:
                    ScaleoutLogger().error(f"Client version: {VERSION} compatibility check failed with Server version: {server_version}. {msg}")
                    return False
                ScaleoutLogger().info("Successfully initialized GRPC connection")
            return True
        except Exception as e:
            ScaleoutLogger().error(f"Could not initialize GRPC connection: {e}")
            return False

    def _send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the server."""
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def _listen_to_task_stream(self, client_id: str) -> None:
        """Listen to the task stream."""
        self.grpc_handler.listen_to_task_stream(client_id=client_id, callback=self._task_stream_callback)

    def default_telemetry_loop(self, update_frequency: float = 5.0) -> None:
        """Send default telemetry data."""
        send_telemetry = True
        while send_telemetry:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            try:
                success = self.log_telemetry(telemetry={"memory_usage": memory_usage, "cpu_usage": cpu_usage})
            except RetryException as e:
                ScaleoutLogger().error(f"Sending telemetry failed: {e}")
                success = False
            if not success:
                ScaleoutLogger().error("Telemetry failed.")
                send_telemetry = False
            time.sleep(update_frequency)

    @contextmanager
    def logging_context(self, context: LoggingContext):
        """Set the logging context."""
        prev_context = self.current_logging_context
        self.current_logging_context = context
        try:
            yield
        finally:
            self.current_logging_context = prev_context

    def _task_stream_callback(self, request: scaleout_msg.TaskRequest) -> None:
        """Handle task stream callbacks."""
        if request.type == TaskType.ModelUpdate.value:
            self.update_local_model(request)
        elif request.type == TaskType.Validation.value:
            self.validate_global_model(request)
        elif request.type == TaskType.Prediction.value:
            self.predict_global_model(request)
        return {}

    def _run_task_callback(self, request: scaleout_msg.TaskRequest) -> Dict:
        if request.type in (t.value for t in TaskType):
            return self._task_stream_callback(request)
        elif TaskType.is_custom_task(request.type):
            return self._handle_custom_task(request)
        else:
            ScaleoutLogger().error(f"Invalid task type: {request.type}")
            raise Exception(f"Invalid task type: {request.type}")

    def _handle_custom_task(self, request: scaleout_msg.TaskRequest) -> Dict:
        if request.type in self.registered_callbacks:
            with self.logging_context(LoggingContext(request=request)):
                params = json.loads(request.data) if request.data else {}
                try:
                    result = self.registered_callbacks[request.type](params)
                except Exception as e:
                    ScaleoutLogger().error(f"Custom task callback failed with exception: {e}")
                    traceback.print_exc()
                    return None
                return result
        else:
            ScaleoutLogger().warning(f"Unknown task type: {request.type}")
            raise UnknownTaskType(f"Unknown task type: {request.type}")

    def update_local_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Update the local model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model_update_id = str(uuid.uuid4())

            tic = time.time()
            in_model = self.get_model_from_combiner(model_id=model_id)

            if in_model is None:
                ScaleoutLogger().error("Could not retrieve model from combiner. Aborting training request.")
                return

            fetch_model_time = time.time() - tic
            ScaleoutLogger().info(f"FETCH_MODEL: {fetch_model_time}")

            if not self.train_callback:
                ScaleoutLogger().error("No train callback set")
                return

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    f"\t Starting processing of training request for model_id {model_id}",
                    log_level=scaleout_msg.LogLevel.INFO,
                    type="MODEL_UPDATE",
                )

            ScaleoutLogger().info(f"Running train callback with model ID: {model_id}")
            client_settings = json.loads(request.data).get("client_settings", {})
            tic = time.time()
            try:
                out_model, meta = self.train_callback(in_model, client_settings)
            except StoppedException:
                return
            except Exception as e:
                ScaleoutLogger().error(f"Train callback failed with exception: {e}")
                traceback.print_exc()
                return
            if out_model is None:
                ScaleoutLogger().error("Train callback returned None model. Aborting training request.")
                return

            meta["processing_time"] = time.time() - tic

            tic = time.time()
            out_model.model_id = model_update_id
            self.send_model_to_combiner(model=out_model)
            meta["upload_model"] = time.time() - tic
            ScaleoutLogger().info("UPLOAD_MODEL: {0}".format(meta["upload_model"]))

            meta["fetch_model"] = fetch_model_time
            meta["config"] = request.data

            self.grpc_handler.send_model_update(
                model_id=model_id,
                model_update_id=model_update_id,
                meta=meta,
                correlation_id=request.correlation_id,
                round_id=request.round_id,
                session_id=request.session_id,
            )

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    "Model update completed.",
                    log_level=scaleout_msg.LogLevel.AUDIT,
                    type="MODEL_UPDATE",
                )

    def validate_global_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Validate the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    f"Processing validate request for model_id {model_id}",
                    log_level=scaleout_msg.LogLevel.INFO,
                    type="MODEL_VALIDATION",
                )

            in_model = self.get_model_from_combiner(model_id=model_id)

            if in_model is None:
                ScaleoutLogger().error("Could not retrieve model from combiner. Aborting validation request.")
                return

            if not self.validate_callback:
                ScaleoutLogger().error("No validate callback set")
                return

            ScaleoutLogger().debug(f"Running validate callback with model ID: {model_id}")
            try:
                metrics = self.validate_callback(in_model)
            except StoppedException:
                return
            except Exception as e:
                ScaleoutLogger().error(f"Validation callback failed with exception: {e}")
                traceback.print_exc()
                return

            if metrics is not None:
                # Send validation
                result: bool = self.grpc_handler.send_model_validation(
                    model_id=request.model_id,
                    metrics=json.dumps(metrics),
                    correlation_id=request.correlation_id,
                    session_id=request.session_id,
                )

                if result and SCALEOUT_CLIENT_STATUS_REPORTING:
                    self.send_status(
                        "Model validation completed.",
                        log_level=scaleout_msg.LogLevel.AUDIT,
                        type="MODEL_VALIDATION",
                    )
                elif SCALEOUT_CLIENT_STATUS_REPORTING:
                    self.send_status(
                        f"Client {self.client_id} failed to complete model validation.",
                        log_level=scaleout_msg.LogLevel.WARNING,
                        type="MODEL_VALIDATION",
                    )

    def predict_global_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Predict using the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model = self.get_model_from_combiner(model_id=model_id)

            if model is None:
                ScaleoutLogger().error("Could not retrieve model from combiner. Aborting prediction request.")
                return

            if not self.predict_callback:
                ScaleoutLogger().error("No predict callback set")
                return

            ScaleoutLogger().info(f"Running predict callback with model ID: {model_id}")
            try:
                prediction = self.predict_callback(model)
            except Exception as e:
                ScaleoutLogger().error(f"Predict callback failed with exception: {e}")
                traceback.print_exc()
                return

            self.grpc_handler.send_model_prediction(
                model_id=request.model_id, prediction_output=json.dumps(prediction), correlation_id=request.correlation_id, session_id=request.session_id
            )

    def log_metric(self, metrics: dict, step: int = None, commit: bool = True, check_task_abort=True, context: LoggingContext = None) -> bool:
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

        message = self.grpc_handler.create_metric_message(
            metrics=metrics,
            model_id=context.model_id,
            step=step,
            round_id=context.round_id,
            session_id=context.session_id,
        )

        success = self.grpc_handler.send_model_metric(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def log_attributes(self, attributes: dict, check_task_abort: bool = True) -> bool:
        """Log the attributes to the server.

        Args:
            attributes (dict): The attributes to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the attributes were logged successfully, False otherwise.

        """
        message = scaleout_msg.AttributeMessage()
        message.client_id = self.client_id
        message.timestamp.GetCurrentTime()

        for key, value in attributes.items():
            message.attributes.add(key=key, value=value)

        success = self.grpc_handler.send_attributes(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def log_telemetry(
        self,
        telemetry: dict,
        check_task_abort: bool = True,
    ) -> bool:
        """Log the telemetry data to the server.

        Args:
            telemetry (dict): The telemetry data to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the telemetry data was logged successfully, False otherwise.

        """
        message = scaleout_msg.TelemetryMessage()
        message.client_id = self.client_id
        message.timestamp.GetCurrentTime()

        for key, value in telemetry.items():
            message.telemetries.add(key=key, value=value)

        success = self.grpc_handler.send_telemetry(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def check_task_abort(self) -> None:
        """Check if the ongoing task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from a thread that do not run the task, this function is a no-op.

        Raises:
            StoppedException: If the task was aborted.

        """
        self.task_receiver.check_abort()

    def set_name(self, name: str) -> None:
        """Set the client name."""
        ScaleoutLogger().info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        ScaleoutLogger().info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    def run(self, with_heartbeat=False, with_polling=True) -> None:
        """Run the client."""

        # Handle SIGTERM for graceful shutdown
        def _handle_sigterm(signum, frame):
            raise GracefulExitException()

        signal.signal(signal.SIGTERM, _handle_sigterm)

        if with_heartbeat:
            threading.Thread(target=self._send_heartbeats, args=(self.name, self.client_id), daemon=True).start()
        if SCALEOUT_CLIENT_SEND_TELEMETRY:
            threading.Thread(target=self.default_telemetry_loop, daemon=True).start()

        try:
            if with_polling:
                self._run_polling_client()
            else:
                self._listen_to_task_stream(client_id=self.client_id)
        except KeyboardInterrupt:
            ScaleoutLogger().info("Client stopped by user.")
        except GracefulExitException:
            ScaleoutLogger().info("Client stopping gracefully.")

    def _run_polling_client(self) -> None:
        self.task_receiver.start()
        ScaleoutLogger().info("Task receiver started.")
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            self.grpc_handler.connect()
        while True:
            try:
                ScaleoutLogger().info("Client is running. Press Ctrl+C to stop.")
                self.task_receiver.wait_on_manager_thread()
                ScaleoutLogger().info("Task manager thread has exited. Stopping client.")
                break
            except GracefulExitException:
                ScaleoutLogger().info("SIGTERM received, shutting down gracefully...")
                if not self.task_receiver.has_current_task():
                    ScaleoutLogger().info("No ongoing task to abort. Exiting...")
                    break
                self.task_receiver.abort_current_task()
                break
            except KeyboardInterrupt:
                ScaleoutLogger().info("KeyboardInterrupt received, aborting current task...")
                if not self.task_receiver.has_current_task():
                    ScaleoutLogger().info("No ongoing task to abort. Exiting client.")
                    break
                self.task_receiver.abort_current_task()
                ScaleoutLogger().info("To completely stop the client, press Ctrl+C again within 5 seconds...")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                ScaleoutLogger().info("Second KeyboardInterrupt received, stopping client immediately...")
                break
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            self.grpc_handler.disconnect()

    def get_model_from_combiner(self, model_id: str) -> ScaleoutModel:
        """Get the model from the combiner."""
        return self.grpc_handler.get_model_from_combiner(model_id=model_id)

    def send_model_to_combiner(self, model: ScaleoutModel) -> scaleout_msg.ModelResponse:
        """Send the model to the combiner."""
        return self.grpc_handler.send_model_to_combiner(model=model)

    def send_status(
        self,
        msg: str,
        log_level: scaleout_msg.LogLevel = scaleout_msg.LogLevel.INFO,
        type: Optional[str] = None,
    ) -> None:
        """Send the status."""
        self.grpc_handler.send_status(msg, log_level, type)
