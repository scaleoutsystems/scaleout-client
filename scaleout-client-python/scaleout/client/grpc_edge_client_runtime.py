"""GrpcEdgeClientRuntime: the default gRPC-backed runtime for :class:`EdgeClient`.

:class:`EdgeClient` owns one ``EdgeClientRuntime`` (a structural protocol)
and delegates connection, task dispatch, transport, and run-loop concerns
to it. ``GrpcEdgeClientRuntime`` is the production implementation backed by
``GrpcHandler`` and ``TaskReceiver``. Alternative runtimes (e.g. test mocks)
need only match the protocol shape — they do not need to inherit from this
class.
"""

import json
import signal
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

import psutil
import requests

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.edge_client import (
    ConnectToApiResult,
    EdgeClient,
    GracefulExitException,
)
from scaleout.client.grpc_handler import GrpcConnectionOptions, GrpcHandler, RetryException
from scaleout.client.logging_context import LoggingContext
from scaleout.client.task_receiver import StoppedException, TaskReceiver, UnknownTaskType
from scaleout.utils.dist import VERSION
from scaleoututil.auth.token_manager import TokenManager
from scaleoututil.config import (
    SCALEOUT_AUTH_SCHEME,
    SCALEOUT_CHECK_COMPATIBILITY,
    SCALEOUT_CLIENT_SEND_TELEMETRY,
    SCALEOUT_CLIENT_STATUS_REPORTING,
    SCALEOUT_CLIENT_TASK_POLLING_INTERVAL,
    SCALEOUT_CONNECT_API_SECURE,
    SCALEOUT_GRACEFUL_CLIENT_CONNECTION,
)
from scaleoututil.grpc.tasktype import TaskType
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.http_status_codes import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_ACCEPTABLE,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_PACKAGE_MISSING,
    HTTP_STATUS_SERVER_ERROR,
    HTTP_STATUS_UNAUTHORIZED,
)
from scaleoututil.utils.model import ScaleoutModel
from scaleoututil.utils.url import assemble_endpoint_url

REQUEST_TIMEOUT = 10  # seconds


class GrpcEdgeClientRuntime:
    """gRPC-backed runtime: connection, task dispatch, and run loop."""

    def __init__(self, client: EdgeClient) -> None:
        """Initialize the runtime with a back-reference to its owning client."""
        self._client = client

        self.grpc_handler: Optional[GrpcHandler] = None
        self.token_manager: Optional[TokenManager] = None

        self.task_receiver = TaskReceiver(self, self._run_task_callback, polling_interval=SCALEOUT_CLIENT_TASK_POLLING_INTERVAL)

    # -- identity proxies ------------------------------------------------------
    # GrpcHandler and TaskReceiver hold a reference to the runtime and read
    # ``client_id`` / ``name`` off it. Both live on EdgeClient; expose them here
    # so the helpers don't need to reach through ``_client`` themselves.

    @property
    def client_id(self) -> Optional[str]:
        return self._client.client_id

    @property
    def name(self) -> Optional[str]:
        return self._client.name

    # -- auth ------------------------------------------------------------------

    def get_access_token(self) -> Optional[str]:
        """Return the current access token, refreshing if needed."""
        if self.token_manager:
            return self.token_manager.get_access_token()
        return None

    def _init_token_manager(self, token: str, url: str, token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None) -> None:
        """Initialize the token manager with the provided token."""
        if self.token_manager is None:
            token_endpoint = assemble_endpoint_url(url, "api/auth", "refresh")
            self.token_manager = TokenManager(refresh_token=token, token_endpoint=token_endpoint, on_token_refresh=token_refresh_callback)

    # -- connection ------------------------------------------------------------

    def connect_to_api(
        self,
        url: str,
        json: Optional[dict] = None,
        token: Optional[str] = None,
        token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the Scaleout API. Accepts a refresh token, instantiates TokenManager, and uses access token."""
        if token:
            self._init_token_manager(token, url, token_refresh_callback)
        current_token = self.get_access_token()

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
                self._client.set_client_id(json_response["client_id"])
                self._client.set_name(json.get("name", json_response["client_id"]))
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

    # -- reporting primitives --------------------------------------------------

    def send_metric(self, metrics: dict, model_id: str, step: int, round_id: str, session_id: str) -> bool:
        """Build and send a model-metric message via gRPC."""
        message = self.grpc_handler.create_metric_message(
            metrics=metrics,
            model_id=model_id,
            step=step,
            round_id=round_id,
            session_id=session_id,
        )
        return self.grpc_handler.send_model_metric(message)

    def send_attributes(self, attributes: dict) -> bool:
        """Build and send an attribute message via gRPC."""
        message = scaleout_msg.AttributeMessage()
        message.client_id = self._client.client_id
        message.timestamp.GetCurrentTime()
        for key, value in attributes.items():
            message.attributes.add(key=key, value=value)
        return self.grpc_handler.send_attributes(message)

    def send_telemetry(self, telemetry: dict) -> bool:
        """Build and send a telemetry message via gRPC."""
        message = scaleout_msg.TelemetryMessage()
        message.client_id = self._client.client_id
        message.timestamp.GetCurrentTime()
        for key, value in telemetry.items():
            message.telemetries.add(key=key, value=value)
        return self.grpc_handler.send_telemetry(message)

    def check_task_abort(self) -> None:
        """Raise StoppedException if the current task has been aborted."""
        self.task_receiver.check_abort()

    # -- runtime loops ---------------------------------------------------------

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
                success = self._client.log_telemetry(telemetry={"memory_usage": memory_usage, "cpu_usage": cpu_usage})
            except RetryException as e:
                ScaleoutLogger().error(f"Sending telemetry failed: {e}")
                success = False
            if not success:
                ScaleoutLogger().error("Telemetry failed.")
                send_telemetry = False
            time.sleep(update_frequency)

    # -- task dispatch ---------------------------------------------------------

    def _task_stream_callback(self, request: scaleout_msg.TaskRequest) -> dict:
        """Handle task stream callbacks."""
        if request.type == TaskType.ModelUpdate.value:
            self.update_local_model(request)
        elif request.type == TaskType.Validation.value:
            self.validate_global_model(request)
        elif request.type == TaskType.StageModel.value:
            self._process_model_stage_request(request)
        elif request.type in TaskType.Inference.value:
            self._process_inference_request(request)
        return {}

    def _run_task_callback(self, request: scaleout_msg.TaskRequest) -> dict:
        if request.type in (t.value for t in TaskType):
            return self._task_stream_callback(request)
        elif TaskType.is_custom_task(request.type):
            return self._handle_custom_task(request)
        else:
            ScaleoutLogger().error(f"Invalid task type: {request.type}")
            raise Exception(f"Invalid task type: {request.type}")

    def _handle_custom_task(self, request: scaleout_msg.TaskRequest) -> dict:
        if request.type in self._client.registered_callbacks:
            with self._client.logging_context(LoggingContext(request=request)):
                request_params = json.loads(request.data) if request.data else {}
                parameters = request_params.get("parameters", {})
                try:
                    result = self._client.registered_callbacks[request.type](parameters)
                except Exception as e:
                    ScaleoutLogger().error(f"Custom task callback failed with exception: {e}")
                    traceback.print_exc()
                    return None
                return result
        else:
            ScaleoutLogger().warning(f"Unknown task type: {request.type}")
            raise UnknownTaskType(f"Unknown task type: {request.type}")

    # -- task handlers ---------------------------------------------------------

    def update_local_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Update the local model."""
        with self._client.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model_update_id = str(uuid.uuid4())

            tic = time.time()
            in_model = self.get_model_from_combiner(model_id=model_id)

            if in_model is None:
                ScaleoutLogger().error("Could not retrieve model from combiner. Aborting training request.")
                return

            fetch_model_time = time.time() - tic
            ScaleoutLogger().info(f"FETCH_MODEL: {fetch_model_time}")

            if not self._client.train_callback:
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
                out_model, meta = self._client.train_callback(in_model, client_settings)
            except StoppedException:
                raise
            except Exception as e:
                ScaleoutLogger().error(f"Train callback failed with exception: {e}")
                traceback.print_exc()
                raise
            if out_model is None:
                ScaleoutLogger().error("Train callback returned None model. Aborting training request.")
                raise Exception("Train callback returned None model.")

            meta["processing_time"] = time.time() - tic

            tic = time.time()
            out_model = out_model.to_builder().set_model_id(model_update_id).build()
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
        with self._client.logging_context(LoggingContext(request=request)):
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

            if not self._client.validate_callback:
                ScaleoutLogger().error("No validate callback set")
                return

            ScaleoutLogger().debug(f"Running validate callback with model ID: {model_id}")
            try:
                metrics = self._client.validate_callback(in_model)
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
                        f"Client {self._client.client_id} failed to complete model validation.",
                        log_level=scaleout_msg.LogLevel.WARNING,
                        type="MODEL_VALIDATION",
                    )

    def _process_model_stage_request(self, task_request: scaleout_msg.TaskRequest) -> None:
        model_id = task_request.model_id
        if not model_id:
            raise ValueError("Model ID is required to stage a model.")
        model = self._client.stage_model(model=model_id)
        if self._client.stage_model_callback is not None:
            self._client.stage_model_callback(model)

    def _process_inference_request(self, task_request: scaleout_msg.TaskRequest) -> None:
        model_id = task_request.model_id
        if not model_id:
            raise ValueError("Model ID is required to run inference.")
        params = json.loads(task_request.data).get("parameters", {})
        return self._client.run_inference(model=model_id, params=params)

    # -- run loop --------------------------------------------------------------

    def run(self, with_heartbeat: bool = False, with_polling: bool = True) -> None:
        """Run the client."""
        # Handle SIGTERM for graceful shutdown
        if threading.current_thread() == threading.main_thread():

            def _handle_sigterm(signum, frame):
                raise GracefulExitException()

            signal.signal(signal.SIGTERM, _handle_sigterm)

        if with_heartbeat:
            threading.Thread(target=self._send_heartbeats, args=(self._client.name, self._client.client_id), daemon=True).start()
        if SCALEOUT_CLIENT_SEND_TELEMETRY:
            threading.Thread(target=self.default_telemetry_loop, daemon=True).start()

        try:
            if with_polling:
                self._run_polling_client()
            else:
                self._listen_to_task_stream(client_id=self._client.client_id)
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
                if not self.task_receiver.has_current_tasks():
                    ScaleoutLogger().info("No ongoing task to abort. Exiting...")
                    break
                self.task_receiver.abort_all_current_tasks()
                break
            except KeyboardInterrupt:
                ScaleoutLogger().info("KeyboardInterrupt received, aborting current task...")
                if not self.task_receiver.has_current_tasks():
                    ScaleoutLogger().info("No ongoing task to abort. Exiting client.")
                    break
                self.task_receiver.abort_all_current_tasks()
                ScaleoutLogger().info("To completely stop the client, press Ctrl+C again within 5 seconds...")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                ScaleoutLogger().info("Second KeyboardInterrupt received, stopping client immediately...")
                break
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            self.grpc_handler.disconnect()

    # -- gRPC passthroughs -----------------------------------------------------

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
