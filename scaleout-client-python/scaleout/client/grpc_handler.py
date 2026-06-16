"""GrpcHandler class for handling GRPC connections and operations."""

import json
import os
import random
import time
from functools import wraps
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from scaleoututil.grpc.clientrequest import ClientRequestType
from scaleout.utils.dist import VERSION
import grpc

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
import scaleoututil.grpc.scaleout_pb2_grpc as rpc
from scaleoututil.config import SCALEOUT_AUTH_SCHEME, SCALEOUT_GRPC_SECURE
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.model import ScaleoutModel

if TYPE_CHECKING:
    from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime  # not-floating-import

# Keepalive settings: these help keep the connection open for long-lived clients
CHUNK_SIZE = 32 * 1024  # 32KB

KEEPALIVE_TIME_MS = 5 * 1000  # send keepalive ping every 5 second
# wait 30 seconds for keepalive ping ack before considering connection dead
KEEPALIVE_TIMEOUT_MS = 30 * 1000
# allow keepalive pings even when there are no RPCs
KEEPALIVE_PERMIT_WITHOUT_CALLS = True

GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
    ("grpc.keepalive_permit_without_calls", KEEPALIVE_PERMIT_WITHOUT_CALLS),
]


def upload_request_generator(model_stream: BytesIO):
    """Generator function for model upload requests for the client

    :param mdl: The model update object.
    :type mdl: BytesIO
    :return: A model update request.
    :rtype: scaleout.FileChunk
    """
    while True:
        b = model_stream.read(CHUNK_SIZE)
        if b:
            yield scaleout_msg.FileChunk(data=b)
        else:
            break


class GrpcAuth(grpc.AuthMetadataPlugin):
    """GRPC authentication plugin."""

    def __init__(self, key_or_callable) -> None:
        """Initialize GrpcAuth with a key or callable that returns a key."""
        if callable(key_or_callable):
            self._key_callable = key_or_callable
            self._key = None
        else:
            self._key = key_or_callable
            self._key_callable = None

    def __call__(self, context: grpc.AuthMetadataContext, callback: grpc.AuthMetadataPluginCallback) -> None:
        """Add authorization metadata to the GRPC call."""
        key = self._key_callable() if self._key_callable else self._key
        # Only add authorization metadata if we have a valid token
        if key is not None:
            callback((("authorization", f"{SCALEOUT_AUTH_SCHEME} {key}"),), None)
        else:
            callback((), None)


class RetryException(Exception):
    pass


class MisconfigurationException(Exception):
    pass


def grpc_retry(
    max_retries: int = 3,
    base_retry_interval: float = 1.0,
    backoff: float = 1.5,
) -> Callable:
    """GRPC retry decorator.


    :param max_retries: The maximum number of retries. -1 means infinite retries.
    :type max_retries: int
    :param retry_interval: The interval between retries in seconds.
    :type retry_interval: float
    :return: The decorated function.
    :rtype: Callable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "GrpcHandler", *args, **kwargs):
            """Wrapper function for retrying GRPC calls."""
            retries = 0
            last_try = time.time()
            backoff_factor = 1.0 / backoff  # so that it is 1.0 on the first try
            while max_retries > retries or max_retries == -1:
                retries += 1
                backoff_factor *= backoff

                # Reset backoff factor if the last try was more than 16 times the retry interval ago
                # This is to prevent the backoff factor from growing too large
                # if the server is down for a long time and then comes back up
                this_try = time.time()
                if this_try - last_try > 16 * base_retry_interval:
                    backoff_factor = 1.0
                last_try = this_try

                retry_interval = base_retry_interval * (backoff_factor + random.uniform(-0.5, 0.5))
                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as e:
                    status_code = e.code()
                    if status_code == grpc.StatusCode.UNAVAILABLE:
                        ScaleoutLogger().warning(f"GRPC ({func.__name__}): Server unavailable. Retrying in {retry_interval:.2f} seconds.")
                        ScaleoutLogger().debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        time.sleep(retry_interval)
                        self._reconnect_channel()
                        continue
                    if status_code == grpc.StatusCode.FAILED_PRECONDITION:
                        ScaleoutLogger().warning(f"GRPC ({func.__name__}): Failed precondition. Retrying in approx {retry_interval:.2f} seconds.")
                        ScaleoutLogger().debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        time.sleep(retry_interval)
                        continue
                    if status_code == grpc.StatusCode.CANCELLED:
                        ScaleoutLogger().warning(f"GRPC ({func.__name__}): Connection cancelled. Retrying in approx {retry_interval:.2f} seconds.")
                        ScaleoutLogger().debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        time.sleep(retry_interval)
                        continue
                    if status_code == grpc.StatusCode.UNKNOWN:
                        details = e.details()
                        if details == "Stream removed":
                            ScaleoutLogger().warning(f"GRPC ({func.__name__}): Stream removed. Retrying in approx {retry_interval:.2f} seconds.")
                            time.sleep(retry_interval)
                            self._reconnect_channel()
                            continue
                        raise e
                    raise e
                except Exception as e:
                    ScaleoutLogger().warning(f"GRPC ({func.__name__}): An unknown error occurred: {e}.")
                    if isinstance(e, ValueError):
                        ScaleoutLogger().warning(f"GRPC ({func.__name__}): Retrying in approx {retry_interval:.2f} seconds.")
                        time.sleep(retry_interval)
                        self._reconnect_channel()
                        continue
                    raise e

            ScaleoutLogger().error(f"GRPC ({func.__name__}): Max retries exceeded.")
            raise RetryException("Max retries exceeded")

        return wrapper

    return decorator


class GrpcHandler:
    """Handler for GRPC connections and operations."""

    def __init__(self, client: "GrpcEdgeClientRuntime", host: str, port: int) -> None:
        """Initialize the GrpcHandler."""
        self.client = client

        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"  # Actively disable fork support in GRPC

        self.host = host
        self.port = port

        self._init_channel(host, port)
        self._init_stubs()

    @property
    def client_id(self) -> str:
        return self.client.client_id

    @property
    def metadata(self) -> list:
        val = [("client", self.client.client_id), ("client-id", self.client.client_id), ("name", self.client.name)]

        # Add authorization token - get fresh token if TokenManager is available
        if hasattr(self.client, "token_manager") and self.client.token_manager:
            current_token = self.client.token_manager.get_access_token()
            val.append(("authorization", f"{SCALEOUT_AUTH_SCHEME} {current_token}"))
        elif hasattr(self, "_auth_token") and self._auth_token:
            val.append(("authorization", f"{SCALEOUT_AUTH_SCHEME} {self._auth_token}"))

        return val

    def _init_stubs(self) -> None:
        """Initialize GRPC stubs."""
        self.combinerStub = rpc.CombinerClientStub(self.channel)
        self.modelStub = rpc.ModelServiceStub(self.channel)

    def _init_channel(self, host: str, port: int) -> None:
        """Initialize the GRPC channel.

        Secure (TLS) by default; controlled by SCALEOUT_GRPC_SECURE rather than the
        port number. Set SCALEOUT_GRPC_SECURE=false for plaintext combiners.
        """
        token = self.client.token_manager.get_access_token() if hasattr(self.client, "token_manager") and self.client.token_manager else None
        if SCALEOUT_GRPC_SECURE:
            self._init_secure_channel(host, port, token)
        else:
            self._init_insecure_channel(host, port, token)

    def _init_secure_channel(self, host: str, port: int, token: str) -> None:
        """Initialize a secure GRPC channel."""
        url = f"{host}:{port}"
        ScaleoutLogger().info(f"Connecting (GRPC) to {url}")

        if os.getenv("SCALEOUT_GRPC_ROOT_CERT_PATH"):
            ScaleoutLogger().info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["SCALEOUT_GRPC_ROOT_CERT_PATH"], "rb") as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            self.channel = grpc.secure_channel(
                f"{host}:{port}",
                credentials,
                options=GRPC_OPTIONS,
            )
            return

        credentials = grpc.ssl_channel_credentials()

        if hasattr(self.client, "token_manager") and self.client.token_manager:
            # The token is attached per call via the metadata property, so it must
            # not also be added to the channel here: doing both sends a duplicate
            # authorization header, which proxies reject with HTTP 400.
            self.channel = grpc.secure_channel(
                f"{host}:{port}",
                credentials,
                options=GRPC_OPTIONS,
            )
            ScaleoutLogger().info("Using TokenManager; token sent via per-call metadata in secure channel")
        elif token:
            # Unreachable today (token is only sourced from token_manager above).
            auth_creds = grpc.metadata_call_credentials(GrpcAuth(token))
            self.channel = grpc.secure_channel(
                f"{host}:{port}",
                grpc.composite_channel_credentials(credentials, auth_creds),
                options=GRPC_OPTIONS,
            )
            ScaleoutLogger().info("Using static token for authentication in secure channel")
        else:
            # No authentication available - create channel without auth credentials
            self.channel = grpc.secure_channel(
                f"{host}:{port}",
                credentials,
                options=GRPC_OPTIONS,
            )
            ScaleoutLogger().info("No authentication token available - connecting without auth credentials")

    def _init_insecure_channel(self, host: str, port: int, token: Optional[str] = None) -> None:
        """Initialize an insecure GRPC channel."""
        url = f"{host}:{port}"
        ScaleoutLogger().info(f"Connecting (GRPC) to {url}")
        self.channel = grpc.insecure_channel(
            url,
            options=GRPC_OPTIONS,
        )
        # Store token to add to metadata in each call
        self._auth_token = token if token else None
        if self._auth_token:
            ScaleoutLogger().info("Token will be added to metadata for authentication.")

    def heartbeat(self) -> scaleout_msg.Response:
        """Send a heartbeat to the combiner.

        :return: Response from the combiner.
        :rtype: scaleout.Response
        """
        heartbeat = scaleout_msg.Heartbeat(client_id=self.client_id)

        response = self.combinerStub.SendHeartbeat(heartbeat, metadata=self.metadata)

        return response

    @grpc_retry(max_retries=-1)
    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the combiner at regular intervals."""
        send_heartbeat = True
        while send_heartbeat:
            response = self.heartbeat()
            time.sleep(update_frequency)
            if isinstance(response, scaleout_msg.Response):
                pass
            else:
                ScaleoutLogger().error("Heartbeat failed.")
                send_heartbeat = False

    @grpc_retry(max_retries=-1)
    def listen_to_task_stream(self, client_id: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to the model update request stream."""
        r = scaleout_msg.ClientAvailableMessage()
        r.client_id = client_id

        ScaleoutLogger().info("Listening to task stream.")
        request: scaleout_msg.TaskRequest
        for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
            ScaleoutLogger().info(f"Received task request of type {request.type} for model_id {request.model_id}")
            callback(request)

    @grpc_retry(max_retries=-1)
    def PollAndReportAsync(self, report: scaleout_msg.ClientReport) -> scaleout_msg.CombinerDirective:
        return self.combinerStub.PollAndReportAsync(report, metadata=self.metadata)

    @grpc_retry(max_retries=5)
    def send_status(
        self,
        msg: str,
        log_level: scaleout_msg.LogLevel = scaleout_msg.LogLevel.INFO,
        type: Optional[str] = None,
    ) -> None:
        """Send status message.

        :param msg: The message to send.
        :type msg: str
        :param log_level: The log level of the message.
        :type log_level: scaleout.LogLevel.INFO, scaleout.LogLevel.WARNING, scaleout.LogLevel.ERROR
        :param type: The type of the message.
        :type type: str
        :param request: The request message.
        :type request: scaleout.Request
        """
        status = scaleout_msg.Status()
        status.timestamp.GetCurrentTime()
        status.client_id = self.client_id
        status.log_level = log_level
        status.status = str(msg)

        if type is not None:
            status.type = type

        ScaleoutLogger().info("Sending status message to combiner.")
        _ = self.combinerStub.SendStatus(status, metadata=self.metadata)

    @grpc_retry(max_retries=5)
    def send_model_metric(self, metric: scaleout_msg.ModelMetric) -> bool:
        """Send a model metric to the combiner."""
        ScaleoutLogger().info("Sending model metric to combiner")
        _ = self.combinerStub.SendModelMetric(metric, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5)
    def send_attributes(self, attribute: scaleout_msg.AttributeMessage) -> bool:
        """Send a attribute message to the combiner."""
        ScaleoutLogger().debug("Sending attributes to combiner.")
        _ = self.combinerStub.SendAttributeMessage(attribute, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5)
    def send_telemetry(self, telemetry: scaleout_msg.TelemetryMessage) -> bool:
        """Send a telemetry message to the combiner."""
        ScaleoutLogger().debug("Sending telemetry to combiner.")
        _ = self.combinerStub.SendTelemetryMessage(telemetry, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1)
    def connect(self) -> scaleout_msg.Response:
        """Connect the client to the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.Connect.value
        ScaleoutLogger().info("Connecting client to combiner.")
        response = self.combinerStub.Announce(request, metadata=self.metadata)
        return response

    def disconnect(self) -> scaleout_msg.Response:
        """Disconnect the client from the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.Disconnect.value
        ScaleoutLogger().info("Disconnecting client from combiner.")
        response = self.combinerStub.Announce(request, metadata=self.metadata)
        return response

    def check_version_compatibility(self) -> Tuple[bool, str, str]:
        """Check version compatibility with the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.VersionCheck.value
        request.parameters = json.dumps({"client-version": VERSION})
        response: scaleout_msg.Response = self.combinerStub.Announce(request, metadata=self.metadata)
        params = json.loads(response.parameters) if response.parameters else {}
        success = params.get("success", False)
        server_version = params.get("version", "")
        msg = response.response
        return success, server_version, msg

    @grpc_retry(max_retries=-1)
    def get_model_from_combiner(self, model_id: str) -> ScaleoutModel:
        """Fetch a model from the assigned combiner.

        Downloads the model update object via a gRPC streaming channel.

        :param id: The id of the model update object.
        :type id: str
        :param client_id: The id of the client.
        :type client_id: str
        :param timeout: The timeout for the request.
        :type timeout: int
        :return: The model update object.
        :rtype: ScaleoutModel
        """
        request = scaleout_msg.ModelRequest(model_id=model_id)

        ScaleoutLogger().info("Downloading model from combiner.")
        part_iterator = self.modelStub.Download(request, metadata=self.metadata)
        model = ScaleoutModel.from_filechunk_stream(part_iterator)
        if model.legacy_source:
            # For legacy models loaded from raw binary streams without embedded metadata, we need to ensure the model_id is set correctly
            model = model.to_builder().set_model_id(model_id).build()  # Ensure model_id is equal to requested model_id for legacy models
        if model.model_id != model_id:
            ScaleoutLogger().error("Model ID mismatch! Expected {}, got {}.".format(model_id, model.model_id))
            raise MisconfigurationException("Model ID mismatch! Expected {}, got {}.".format(model_id, model.model_id))
        metadata = dict(part_iterator.trailing_metadata())
        server_checksum = metadata.get("checksum")
        if server_checksum:
            if not model.verify_checksum(server_checksum):
                ScaleoutLogger().error("Checksum mismatch! File is corrupted!")
                raise ValueError("Checksum mismatch! File is corrupted!")
            else:
                ScaleoutLogger().info("Checksum match! File is valid!")
        return model

    @grpc_retry(max_retries=-1)
    def send_model_to_combiner(self, model: ScaleoutModel) -> scaleout_msg.ModelResponse:
        """Send a model update to the assigned combiner.

        Uploads the model updated object via a gRPC streaming channel, Upload.

        :param model: The model update object.
        :type model: BytesIO
        :param id: The id of the model update object.
        :type id: str
        :return: The model upload response.
        :rtype: ModelResponse
        """
        file_checksum = model.checksum
        model_id = model.model_id

        ScaleoutLogger().info("Uploading model to combiner.")
        metadata = [*self.metadata, ("model-id", model_id), ("checksum", file_checksum)]
        result: scaleout_msg.ModelResponse = self.modelStub.Upload(model.get_filechunk_stream(), metadata=metadata)

        return result

    def create_metric_message(self, metrics: dict, step: int, model_id: str, session_id: str, round_id: str) -> scaleout_msg.ModelMetric:
        """Create a metric message."""
        metric = scaleout_msg.ModelMetric()
        metric.client_id = self.client_id
        metric.model_id = model_id
        if step is not None:
            metric.step.value = step
        metric.session_id = session_id
        metric.round_id = round_id
        metric.timestamp.GetCurrentTime()
        for key, value in metrics.items():
            metric.metrics.add(key=key, value=value)
        return metric

    def send_model_update(self, model_id: str, model_update_id: str, correlation_id: str, round_id: str, session_id: str, meta: dict) -> bool:
        """Send a model update to the combiner."""
        update = scaleout_msg.ModelUpdate()
        update.client_id = self.client_id
        update.model_id = model_id
        update.model_update_id = model_update_id
        update.correlation_id = correlation_id
        update.round_id = round_id
        update.session_id = session_id
        update.timestamp.GetCurrentTime()
        update.meta = json.dumps(meta)
        return self._send_model_update(update)

    @grpc_retry(max_retries=-1)
    def _send_model_update(self, update: scaleout_msg.ModelUpdate) -> bool:
        """Send a model update to the combiner."""
        ScaleoutLogger().info("Sending model update to combiner.")
        _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        return True

    def send_model_validation(
        self,
        model_id: str,
        metrics: str,
        correlation_id: str,
        session_id: str,
    ) -> bool:
        """Send a model validation to the combiner."""
        validation = scaleout_msg.ModelValidation()
        validation.client_id = self.client_id
        validation.model_id = model_id
        validation.data = metrics
        validation.timestamp.GetCurrentTime()
        validation.correlation_id = correlation_id
        validation.session_id = session_id

        return self._send_model_validation(validation)

    @grpc_retry(max_retries=-1)
    def _send_model_validation(self, validation: scaleout_msg.ModelValidation) -> bool:
        """Send a model validation to the combiner."""
        ScaleoutLogger().info("Sending model validation to combiner.")
        _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        return True

    def _disconnect_channel(self) -> None:
        """Disconnect from the combiner."""
        self.channel.close()
        ScaleoutLogger().info("GRPC channel closed.")

    def _reconnect_channel(self) -> None:
        """Reconnect to the combiner."""
        self._disconnect_channel()
        self._init_channel(self.host, self.port)
        self._init_stubs()
        ScaleoutLogger().debug("GRPC channel reconnected.")


class GrpcConnectionOptions:
    """Options for configuring the GRPC connection."""

    def __init__(self, host: str, port: int, internal_hostname: str = "", package: str = "", helper_type: str = "") -> None:
        """Initialize GrpcConnectionOptions."""
        self.host = host
        self.internal_hostname = internal_hostname
        self.package = package
        self.port = port
        self.helper_type = helper_type
        if not host:
            self.host = internal_hostname

    @classmethod
    def from_dict(cls, config: dict) -> "GrpcConnectionOptions":
        """Create a GrpcConnectionOptions instance from a JSON string."""
        return cls(
            host=config.get("public_hostname", ""),
            port=config.get("port", 0),
            internal_hostname=config.get("internal_hostname", ""),
            package=config.get("package", ""),
            helper_type=config.get("helper_type", ""),
        )
