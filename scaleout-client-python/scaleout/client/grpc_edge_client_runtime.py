"""GrpcEdgeClientRuntime: the default gRPC-backed runtime for :class:`EdgeClient`.

:class:`EdgeClient` owns one ``EdgeClientRuntime`` (a structural protocol)
and delegates connection, task dispatch, transport, and run-loop concerns
to it. ``GrpcEdgeClientRuntime`` is the production implementation backed by
``GrpcHandler`` and ``TaskReceiver``. Alternative runtimes (e.g. test mocks)
need only match the protocol shape — they do not need to inherit from this
class.
"""

import json
import random
import signal
import threading
import time
import traceback
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

import grpc
import requests

from scaleout.client.package_runtime import restart_with_venv_and_package, valid_env_for_restart
import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleout.client.edge_client import (
    ConnectToApiResult,
    EdgeClient,
    EdgeClientRuntime,
    GracefulExitException,
)
from scaleout.client.grpc_handler import GrpcConnectionOptions, GrpcHandler
from scaleout.client.logging_context import LoggingContext
from scaleout.client.grpc_transport import GrpcTransportAdapter, grpc_envelope
from scaleout.client.inference_context import InferenceContext
from scaleout.client.streaming_record import GrpcRoutingTransportAdapter
from scaleout.client.task_receiver import StoppedException, Task, TaskReceiver, UnknownTaskType
from scaleout.utils.dist import VERSION
from scaleoututil.auth.login import Login
from scaleoututil.auth.token_manager import TokenManager
from scaleoututil.queue import (
    ClassOrder,
    DEFAULT_CLASS_NAME,
    DecayPolicy,
    InMemoryBackend,
    LinkQuality,
    PriorityClass,
    PriorityQueue,
)
from scaleoututil.config import (
    SCALEOUT_AUTH_SCHEME,
    SCALEOUT_CHECK_COMPATIBILITY,
    SCALEOUT_CLIENT_LEGACY_TELEMETRY,
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
    HTTP_STATUS_TOO_MANY_REQUESTS,
    HTTP_STATUS_UNAUTHORIZED,
)
from scaleoututil.utils.model import ScaleoutModel
from scaleoututil.utils.url import assemble_endpoint_url

REQUEST_TIMEOUT = 10  # seconds

# Retry policy for connect_to_api. Transient failures (network errors,
# timeouts, HTTP 429, and 5xx) are retried with exponential backoff plus
# jitter. Jitter matters when many clients are launched at once: without it
# they would retry in lockstep and keep colliding on the same overloaded
# endpoint (thundering herd).
CONNECT_MAX_ATTEMPTS = 5
CONNECT_BACKOFF_BASE = 1.0  # seconds; doubled each attempt
CONNECT_BACKOFF_CAP = 30.0  # seconds; upper bound on a single backoff


_DEFAULT_PRIORITY_CLASSES: list[PriorityClass] = [
    PriorityClass(name="alert", priority=80, order=ClassOrder.FIFO),
    PriorityClass(name="model_update", priority=70, order=ClassOrder.FIFO),
    PriorityClass(
        name="telemetry",
        priority=40,
        order=ClassOrder.LIFO,
        decay=DecayPolicy(demote_after=(300.0, "backlog")),
    ),
    PriorityClass(
        name="inference",
        priority=30,
        order=ClassOrder.LIFO,
        decay=DecayPolicy(demote_after=(300.0, "backlog")),
    ),
    PriorityClass(name="artifact", priority=20, order=ClassOrder.FIFO),
    PriorityClass(name="backlog", priority=10, order=ClassOrder.FIFO),
]


class GrpcEdgeClientRuntime(EdgeClientRuntime):
    """gRPC-backed runtime: connection, task dispatch, and run loop."""

    def __init__(self) -> None:
        """Initialize the runtime"""
        self._client = None

        self.grpc_handler: Optional[GrpcHandler] = None
        self._login: Optional[Login] = None
        self.token_manager: Optional[TokenManager] = None
        self.queue: Optional[PriorityQueue] = None
        self._backend: Optional[InMemoryBackend] = None
        self._routing_adapter: Optional[GrpcRoutingTransportAdapter] = None
        self._use_legacy_telemetry: bool = SCALEOUT_CLIENT_LEGACY_TELEMETRY
        self._backlog_report_stop = threading.Event()

        self.task_receiver = TaskReceiver(self, self._run_task_callback, polling_interval=SCALEOUT_CLIENT_TASK_POLLING_INTERVAL)

        self._restart_callback = None

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

    @property
    def link_quality(self) -> LinkQuality:
        if self.grpc_handler is not None:
            return self.grpc_handler.link_quality_estimator.quality
        return LinkQuality.HEALTHY

    # -- init ------------------------------------------------------------------

    def set_client(self, client: EdgeClient):
        self._client = client

    # -- auth ------------------------------------------------------------------

    def get_access_token(self) -> Optional[str]:
        """Return the current access token, refreshing if needed."""
        if self._login:
            return self._login.get_access_token()
        return None

    def _get_current_token(self) -> Optional[str]:
        """Alias used by GrpcHandler for dynamic token refresh."""
        return self.get_access_token()

    def _init_login(self, token: str, url: str) -> None:
        """Initialise Login with the provided credential (enrollment, client_refresh, or admin token)."""
        if self._login is None:
            self._login = Login(server_url=url, credential=token)

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
            self._init_login(token, url)
        try:
            current_token = self.get_access_token()
        except RuntimeError as e:
            ScaleoutLogger().error(f"Connect to Scaleout API - Failed to obtain access token: {e}")
            ScaleoutLogger().info("Hint: The refresh token may have expired or been revoked. Re-enroll the client to get a new token.")
            return ConnectToApiResult.UnAuthorized, str(e)

        url_endpoint = assemble_endpoint_url(url, "api/v1/clients/add")
        ScaleoutLogger().info(f"Connecting to API endpoint: {url_endpoint}")

        if SCALEOUT_CHECK_COMPATIBILITY:
            json["client_version"] = VERSION

        last_error = "Unknown error"
        for attempt in range(CONNECT_MAX_ATTEMPTS):
            if attempt > 0:
                delay = self._connect_backoff(attempt)
                ScaleoutLogger().warning(
                    f"Connect to Scaleout API - retrying in {delay:.1f}s (attempt {attempt + 1}/{CONNECT_MAX_ATTEMPTS}); last error: {last_error}"
                )
                time.sleep(delay)

            try:
                response = requests.post(
                    url=url_endpoint,
                    json=json,
                    allow_redirects=True,
                    headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {current_token}"},
                    timeout=REQUEST_TIMEOUT,
                    verify=SCALEOUT_CONNECT_API_SECURE,
                )
            except Exception as e:
                # Network error / timeout / connection reset: transient, retry.
                last_error = str(e)
                continue

            status_code = response.status_code

            if status_code == HTTP_STATUS_OK:
                ScaleoutLogger().info("Connect to Scaleout API - Client assigned to controller")
                json_response = response.json()
                self._client.set_client_id(json_response["client_id"])
                self._client.set_name(json.get("name", json_response["client_id"]))
                combiner_config = GrpcConnectionOptions.from_dict(json_response)
                return ConnectToApiResult.Assigned, combiner_config

            if status_code == HTTP_STATUS_PACKAGE_MISSING:
                json_response = response.json()
                ScaleoutLogger().info("Connect to Scaleout API - Remote compute package missing.")
                return ConnectToApiResult.ComputePackageMissing, json_response

            if status_code == HTTP_STATUS_UNAUTHORIZED:
                ScaleoutLogger().error("Connect to Scaleout API - Unauthorized")
                return ConnectToApiResult.UnAuthorized, "Unauthorized"

            if status_code in (HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_NOT_ACCEPTABLE):
                msg = self._response_message(response, default="Unknown error")
                ScaleoutLogger().error(f"Connect to Scaleout API - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if status_code == HTTP_STATUS_NOT_FOUND:
                ScaleoutLogger().error("Connect to Scaleout API - Incorrect URL")
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"

            if status_code == HTTP_STATUS_TOO_MANY_REQUESTS or status_code >= HTTP_STATUS_SERVER_ERROR:
                # Overloaded / unavailable upstream (429, 5xx): transient, retry.
                last_error = f"HTTP {status_code}: {self._response_message(response, default='server busy or unavailable')}"
                continue

            # Any other unhandled status code is terminal — don't loop forever.
            last_error = f"Unexpected HTTP {status_code}: {self._response_message(response, default='unhandled response')}"
            ScaleoutLogger().error(f"Connect to Scaleout API - {last_error}")
            return ConnectToApiResult.UnknownError, last_error

        ScaleoutLogger().error(f"Connect to Scaleout API - giving up after {CONNECT_MAX_ATTEMPTS} attempts; last error: {last_error}")
        return ConnectToApiResult.UnknownError, last_error

    @staticmethod
    def _connect_backoff(attempt: int) -> float:
        """Exponential backoff with equal jitter for the connect retry loop.

        ``attempt`` is the 1-based retry index. Returns a delay in
        ``[exp/2, exp]`` where ``exp = min(cap, base * 2**(attempt-1))`` — the
        jitter spreads simultaneously-launched clients apart while guaranteeing
        the wait still grows with each attempt.
        """
        exp = min(CONNECT_BACKOFF_CAP, CONNECT_BACKOFF_BASE * (2 ** (attempt - 1)))
        return exp / 2.0 + random.uniform(0.0, exp / 2.0)

    @staticmethod
    def _response_message(response: requests.Response, default: str) -> str:
        """Best-effort extraction of a ``message`` field from a JSON response body."""
        try:
            return response.json().get("message", default)
        except Exception:
            return default

    def init_grpchandler(
        self,
        config: GrpcConnectionOptions,
        token: Optional[str] = None,
        url: Optional[str] = None,
        token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> bool:
        """Initialize the GRPC handler. Accepts a refresh token, instantiates TokenManager, and uses access token."""
        if token and url:
            self._init_login(token, url)
        try:
            self.grpc_handler = GrpcHandler(self, host=config.host, port=config.port)

            self._backend = InMemoryBackend(classes=_DEFAULT_PRIORITY_CLASSES)
            self._routing_adapter = GrpcRoutingTransportAdapter(
                handler=self.grpc_handler,
                unary=GrpcTransportAdapter(self.grpc_handler),
                mark_acked=self._backend.mark_acked,
                requeue_inflight=self._backend.requeue_inflight,
                use_legacy_telemetry=self._use_legacy_telemetry,
            )
            self.queue = PriorityQueue(
                transport=self._routing_adapter,
                backend=self._backend,
            )
            self.queue.start()
            self.grpc_handler.on_reconnect = self.queue.wake_drainer
            ScaleoutLogger().info(f"Priority queue started with classes: {sorted(self._backend.stats().keys())}")

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
        """Enqueue a model-metric message. True means accepted into the queue, not delivered."""
        message = self.grpc_handler.create_metric_message(
            metrics=metrics,
            model_id=model_id,
            step=step,
            round_id=round_id,
            session_id=session_id,
        )
        self.queue.enqueue(grpc_envelope(message, "artifact"))
        return True

    def send_attributes(self, attributes: dict, priority: Optional[str] = None) -> bool:
        """Enqueue attribute records. True means accepted into the queue, not delivered."""
        _priority = priority or "artifact"
        for key, value in attributes.items():
            record = scaleout_msg.AttributeRecord()
            record.client_id = self._client.client_id
            record.timestamp.GetCurrentTime()
            record.key = key
            record.payload = json.dumps(value if isinstance(value, dict) else {"value": value})
            self.queue.enqueue(grpc_envelope(record, _priority))
        return True

    def send_telemetry(
        self,
        key: Optional[str] = None,
        payload: Optional[dict] = None,
        telemetry: Optional[dict] = None,
        priority: Optional[str] = None,
    ) -> bool:
        """Enqueue a telemetry observation. True means accepted into the queue, not delivered.

        Intended usage emits one ``TelemetryRecord`` per call::

            send_telemetry(key="loss", payload={"value": 0.5, "step": 12})

        Legacy form (``telemetry={key: value, ...}``) still works for back-compat
        and is the only accepted shape under ``SCALEOUT_CLIENT_LEGACY_TELEMETRY=true``;
        in record mode it emits a warning and expands to one record per pair.
        Existing positional callers passing a dict (``send_telemetry({"loss": 0.5})``)
        are accepted and re-routed to the legacy ``telemetry`` slot.
        """
        _priority = priority or "telemetry"

        # Back-compat: existing positional callers pass a dict where `key` now lives.
        if isinstance(key, dict):
            if telemetry is not None:
                ScaleoutLogger().warning("send_telemetry: both positional dict and 'telemetry=' supplied; using positional")
            telemetry = key
            key = None

        if telemetry is None and key is None:
            ScaleoutLogger().warning("send_telemetry: called without 'key' or 'telemetry'")
            return False

        if self._use_legacy_telemetry:
            if key is not None or payload is not None:
                ScaleoutLogger().warning("send_telemetry: called with 'key' and 'payload' in legacy mode")
            message = scaleout_msg.TelemetryMessage()
            message.client_id = self._client.client_id
            message.timestamp.GetCurrentTime()
            if telemetry is not None:
                for k, v in telemetry.items():
                    message.telemetries.add(key=k, value=v)
            self.queue.enqueue(grpc_envelope(message, _priority))
            return True

        if telemetry is not None:
            warnings.warn(
                "send_telemetry: passing 'telemetry' as a dict is deprecated; use 'key' (+ 'payload') instead",
                DeprecationWarning,
                stacklevel=2,
            )
            ScaleoutLogger().warning("send_telemetry: the 'telemetry' dict argument is legacy; pass 'key' (+ 'payload') instead")
            for k, v in telemetry.items():
                record = scaleout_msg.TelemetryRecord()
                record.telemetry_id = str(uuid.uuid4())
                record.client_id = self._client.client_id
                record.timestamp.GetCurrentTime()
                record.key = k
                record.payload = json.dumps({"value": v, **(payload or {})})
                self.queue.enqueue(grpc_envelope(record, _priority))
        if key is not None:
            record = scaleout_msg.TelemetryRecord()
            record.telemetry_id = str(uuid.uuid4())
            record.client_id = self._client.client_id
            record.timestamp.GetCurrentTime()
            record.key = key
            record.payload = json.dumps(payload or {})
            self.queue.enqueue(grpc_envelope(record, _priority))
        return True

    def send_inference_result(self, key: str, payload: dict, inference_id: str, model_id: str, priority: Optional[str] = None) -> bool:
        """Enqueue an InferenceResult at inference priority (or the given priority class)."""
        _priority = priority or "inference"
        proto = scaleout_msg.InferenceResult()
        proto.inference_result_id = str(uuid.uuid4())
        proto.node_id = self._client.client_id
        proto.timestamp.GetCurrentTime()
        proto.key = key
        proto.payload = json.dumps(payload)
        proto.inference_id = inference_id
        proto.model_id = model_id
        self.queue.enqueue(grpc_envelope(proto, _priority))
        return True

    def list_priority_classes(self) -> list[PriorityClass]:
        """Return registered priority classes, excluding the internal default catch-all."""
        if self._backend is None:
            return []
        with self._backend._lock:
            return [cls for name, cls in self._backend._classes.items() if name != DEFAULT_CLASS_NAME]

    def add_priority_class(self, cls: PriorityClass) -> None:
        """Register a new priority class."""
        if self._backend is None:
            raise RuntimeError("queue not initialised yet")
        self._backend.add_class(cls)

    def remove_priority_class(self, name: str, force: bool = False) -> int:
        """Remove a priority class. Returns the number of records migrated to the default."""
        if self._backend is None:
            raise RuntimeError("queue not initialised yet")
        return self._backend.remove_class(name, force)

    def replace_priority_class(self, cls: PriorityClass) -> None:
        """Replace an existing priority class definition in-place."""
        if self._backend is None:
            raise RuntimeError("queue not initialised yet")
        self._backend.replace_class(cls)

    def check_task_abort(self) -> None:
        """Raise StoppedException if the current task has been aborted."""
        self.task_receiver.check_abort()

    # -- runtime loops ---------------------------------------------------------

    def _listen_to_task_stream(self, client_id: str) -> None:
        """Listen to the task stream."""
        self.grpc_handler.listen_to_task_stream(client_id=client_id, callback=self._task_stream_callback)

    # -- task dispatch ---------------------------------------------------------

    def _task_stream_callback(self, request: scaleout_msg.TaskRequest) -> dict:
        """Handle task stream callbacks."""
        if request.type == TaskType.ModelUpdate.value:
            self.update_local_model(request)
        elif request.type == TaskType.Validation.value:
            self.validate_global_model(request)
        elif request.type == TaskType.StageModel.value:
            self._process_model_stage_request(request)
        elif request.type == TaskType.Inference.value:
            self._process_inference_request(request)
        elif request.type == TaskType.RestartClient.value:
            self._restart_client(request)
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

            num_examples = meta.get("training_metadata", {}).get("num_examples", 0)
            if not isinstance(num_examples, (int, float)) or num_examples <= 0:
                raise ValueError(f"Train callback must return num_examples > 0 in training_metadata, got: {num_examples!r}")

            meta["processing_time"] = time.time() - tic

            tic = time.time()
            out_model = out_model.to_builder().set_model_id(model_update_id).build()
            self.send_model_to_combiner(model=out_model)
            meta["upload_model"] = time.time() - tic
            ScaleoutLogger().info("UPLOAD_MODEL: {0}".format(meta["upload_model"]))

            meta["fetch_model"] = fetch_model_time
            meta["config"] = request.data

            update = scaleout_msg.ModelUpdate()
            update.client_id = self._client.client_id
            update.model_id = model_id
            update.model_update_id = model_update_id
            update.correlation_id = request.correlation_id
            update.round_id = request.round_id
            update.session_id = request.session_id
            update.timestamp.GetCurrentTime()
            update.meta = json.dumps(meta)
            self.queue.enqueue(grpc_envelope(update, "model_update"))

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
                validation = scaleout_msg.ModelValidation()
                validation.client_id = self._client.client_id
                validation.model_id = request.model_id
                validation.data = json.dumps(metrics)
                validation.timestamp.GetCurrentTime()
                validation.correlation_id = request.correlation_id
                validation.session_id = request.session_id
                self.queue.enqueue(grpc_envelope(validation, "artifact"))

                if SCALEOUT_CLIENT_STATUS_REPORTING:
                    self.send_status(
                        "Model validation completed.",
                        log_level=scaleout_msg.LogLevel.AUDIT,
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
        inference_id = task_request.correlation_id
        ctx = InferenceContext(inference_id=inference_id, model_id=model_id)
        with self._client.inference_context(ctx):
            return self._client.run_inference(model=model_id, params=params)

    def _restart_client(self, task_request: scaleout_msg.TaskRequest) -> None:
        params = json.loads(task_request.data).get("parameters", {}) if task_request.data else {}
        abort_ongoing = params.get("abort_ongoing_tasks", False)
        force = params.get("force", False)

        if not valid_env_for_restart():
            ScaleoutLogger().error("Client not started with valid configuration, can not update compute package")
            raise Exception("Client not started with valid configuration, can not restart")

        task = self.task_receiver.get_current_task()
        if force:
            threading.Thread(target=self._restart, kwargs={"force": True}).start()
            return {}
        elif abort_ongoing:
            self.task_receiver.stop_recieving_new_tasks()
            ScaleoutLogger().info("Aborting current ongoing tasks before updating compute package")
            threading.Thread(target=self._restart, kwargs={"task": task}).start()
            return {}
        elif self.task_receiver.has_other_tasks():
            ScaleoutLogger().error("Client has other tasks running, can not update compute package")
            raise Exception("Client has other tasks running, cannot restart")
        else:
            self.task_receiver.stop_recieving_new_tasks()
            ScaleoutLogger().info("No ongoing tasks, updating compute package")
            threading.Thread(target=self._restart, kwargs={"task": task, "force": True}).start()
            return {}

    def _restart(self, task: Task = None, force: bool = False):
        if task is not None:
            # Wait until instructing task is completed
            time.sleep(1)
            ScaleoutLogger().info("Waiting for update task to be reported...")
            while self.task_receiver.is_task_running(task):
                time.sleep(1)
        if not force:
            # If not force, abort other tasks
            if self.task_receiver.has_current_tasks():
                ScaleoutLogger().info("Aborting and waiting on current tasks to report...")
                self.task_receiver.abort_all_current_tasks()
                while self.task_receiver.has_current_tasks():
                    time.sleep(1)

        # Set restart callback to be called on main thread
        def _restart_call():
            # Use current venv and let the package manager handle the venv selection with another restart
            restart_with_venv_and_package()

        self._restart_callback = _restart_call
        self.task_receiver.stop()

    # -- run loop --------------------------------------------------------------

    def _backlog_report_loop(self, interval_s: float = 30.0) -> None:
        """Periodically send queue pressure snapshot to the combiner via Announce."""
        while not self._backlog_report_stop.wait(interval_s):
            if self.grpc_handler is None or self._backend is None:
                continue
            try:
                stats = self._backend.stats()
                report = scaleout_msg.BacklogReport(
                    client_id=self.client_id,
                    backlog_total=sum(s.depth for s in stats.values()),
                    backlog_dropped_total=sum(s.dropped for s in stats.values()),
                )
                oldest = self._backend.oldest_pending_age_ms()
                if oldest is not None:
                    report.backlog_oldest_age_ms = oldest
                self.grpc_handler.send_backlog_report(report)
            except grpc.RpcError as e:
                if e.code() in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.UNIMPLEMENTED):
                    ScaleoutLogger().warning(f"Backlog report not accepted by server ({e.code().name}: {e.details()}); stopping backlog reports.")
                    break
                ScaleoutLogger().debug(f"Backlog report failed: {e}")
            except Exception as e:
                ScaleoutLogger().error(f"Backlog report failed: {e}")
                break

    def run(self, with_heartbeat: bool = True, with_polling: bool = True) -> None:
        """Run the client."""
        # Handle SIGTERM for graceful shutdown
        if threading.current_thread() == threading.main_thread():

            def _handle_sigterm(signum, frame):
                raise GracefulExitException()

            signal.signal(signal.SIGTERM, _handle_sigterm)
        if SCALEOUT_CLIENT_SEND_TELEMETRY:
            threading.Thread(target=self._client._default_telemetry_loop, daemon=True).start()

        if with_heartbeat and self.grpc_handler is not None:
            self.grpc_handler.start_link_quality_monitor()
            self._backlog_report_stop.clear()
            threading.Thread(target=self._backlog_report_loop, name="backlog-report", daemon=True).start()

        try:
            if with_polling:
                self._run_polling_client()
            else:
                self._listen_to_task_stream(client_id=self._client.client_id)
        except KeyboardInterrupt:
            ScaleoutLogger().info("Client stopped by user.")
        except GracefulExitException:
            ScaleoutLogger().info("Client stopping gracefully.")
        finally:
            self._shutdown()

        # Shutting down grpc channel
        self.grpc_handler.channel.close()
        # Looking for restart callback
        if self._restart_callback:
            self._restart_callback()

    def _run_polling_client(self) -> None:
        self.task_receiver.start()
        ScaleoutLogger().info("Task receiver started.")
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            try:
                self.grpc_handler.send_connect()
            except grpc.RpcError as e:
                ScaleoutLogger().error(f"Connect failed: {e.code()} — {e.details()}")
                ScaleoutLogger().error("Stopping client")
                return
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
            try:
                self.grpc_handler.send_disconnect()
            except grpc.RpcError as e:
                ScaleoutLogger().warning(f"Disconnect failed (ignored): {e.code()} — {e.details()}")

    def _shutdown(self) -> None:
        """Stop telemetry, then drain the queue.

        Order matters: stop the telemetry producer before draining so the
        queue's final records can leave over a still-live channel.
        Idempotent so it can be invoked from multiple exit paths.
        """
        self._client.stop_default_telemetry_loop()
        self._backlog_report_stop.set()
        if self.grpc_handler is not None:
            self.grpc_handler.stop_link_quality_monitor(timeout=5.0)
        # Close streaming adapters before stopping the queue: stopping the queue
        # first would orphan the adapter mid-send.
        if self._routing_adapter is not None:
            self._routing_adapter.close()
            self._routing_adapter = None
        if self.queue is not None:
            self.queue.stop(timeout=10.0)
            self.queue = None
            ScaleoutLogger().info("Queue stopped.")

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
        """Enqueue a status message. Accepted into the queue, not delivered synchronously."""
        status = scaleout_msg.Status()
        status.timestamp.GetCurrentTime()
        status.client_id = self._client.client_id
        status.log_level = log_level
        status.status = str(msg)
        if type is not None:
            status.type = type
        self.queue.enqueue(grpc_envelope(status, "artifact"))
