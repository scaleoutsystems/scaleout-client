import inspect
import os
import re
from datetime import datetime
from typing import Callable, Dict, Optional
import uuid

import requests

from scaleoututil.auth.token_manager import TokenManager
from scaleoututil.auth.token_cache import TokenCache
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.serverfunctions.serverfunctionsbase import ServerFunctionsBase
from scaleoututil.utils.url import parse_url


class Scaleout:
    """Python client for interacting with the statestore and controller.

    :param host: The host of the api server.
    :type host: str
    :param port: The port of the api server.
    :type port: int
    :param secure: Whether to use https.
    :type secure: bool
    :param verify: Whether to verify the server certificate.
    :type verify: bool
    """

    def __init__(
        self,
        host: str,
        port: int = None,
        secure: Optional[bool] = None,
        verify: bool = True,
        token: str = None,
        auth_scheme: str = None,
        token_endpoint: str = None,
        access_token_provider: Optional[Callable[[], Optional[str]]] = None,
    ):
        """Initialize the Scaleout client.

        :param host: The host of the api server.
        :type host: str
        :param port: The port of the api server.
        :type port: int
        :param secure: Whether to use https.
        :type secure: bool
        :param verify: Whether to verify the server SSL certificate (default: True). Set to False for development with self-signed certificates.
        :type verify: bool
        :param token: Refresh token for automatic token management. Can also be set via SCALEOUT_AUTH_TOKEN env var.
        :type token: str
        :param auth_scheme: Authorization scheme (e.g., 'Bearer'). Defaults to 'Bearer'.
        :type auth_scheme: str
        :param token_endpoint: Token refresh endpoint URL. If not provided, will be constructed from host.
        :type token_endpoint: str
        """
        _protocol, _host, _port, _ = parse_url(host)
        if _protocol is not None and secure is not None:
            ScaleoutLogger().warning("Both protocol in host and secure parameter provided. Using protocol from host.")
        if _protocol is not None:
            if _protocol == "https":
                secure = True
            else:
                secure = False
        host = _host

        if _port is not None and port is not None and _port != port:
            ScaleoutLogger().warning(f"Both port in host URL (:{_port}) and port parameter ({port}) provided. Using port parameter: {port}")
        port = port if port else _port

        if secure is None:
            if port == 443:
                secure = True
                ScaleoutLogger().debug("Port 443 detected, automatically using HTTPS. To use HTTP explicitly, set secure=False.")
            else:
                secure = False

        if port is None:
            port = 443 if secure else 80
            ScaleoutLogger().debug(f"No port specified, using default port {port} for {'HTTPS' if secure else 'HTTP'}.")

        self.host = host
        self.port = port
        self.secure = secure
        self.verify = verify
        self.access_token_provider = access_token_provider
        self.auth_scheme = auth_scheme or os.environ.get("SCALEOUT_AUTH_SCHEME", "Bearer")
        self.token_manager: Optional[TokenManager] = None
        self.headers = {}

        # Get token from args or env
        if not token:
            token = os.environ.get("SCALEOUT_AUTH_TOKEN", None)

        safe_host = re.sub(r"[^\w.\-]", "_", host)
        token_cache_id = f"api-client-{safe_host}-{port}"
        token_cache = TokenCache(cache_id=token_cache_id, cache_dir=os.environ.get("SCALEOUT_TOKEN_CACHE_DIR", None))
        if not token and token_cache.exists():
            cached_data = token_cache.load()
            if cached_data and cached_data.get("refresh_token"):
                token = cached_data["refresh_token"]
                ScaleoutLogger().info(f"Loaded refresh token from cache: {token_cache.cache_file}")

        # Clean token if it has scheme prefix
        if token and " " in token:
            token = token.split()[1]

        # Initialize TokenManager with refresh token
        if token:
            # Construct token endpoint if not provided
            if not token_endpoint:
                token_endpoint = self._get_url("api/auth/refresh")

            # Create token refresh callback to save tokens to cache
            def on_token_refresh(access_token: str, refresh_token: str, expires_at: datetime) -> None:
                """Callback to save tokens when they are refreshed."""
                try:
                    token_cache.save(access_token, refresh_token, expires_at)
                    ScaleoutLogger().debug(f"API client tokens updated in cache: {token_cache.cache_file}")
                except Exception as e:
                    ScaleoutLogger().warning(f"Failed to save API client tokens to cache: {e}")

            # TokenManager will automatically fetch the first access token using the refresh token
            try:
                self.token_manager = TokenManager(
                    refresh_token=token,
                    token_endpoint=token_endpoint,
                    verify_ssl=verify,
                    role="admin",
                    on_token_refresh=on_token_refresh,
                )
            except (RuntimeError, requests.exceptions.RequestException) as e:
                ScaleoutLogger().warning(
                    f"Token authentication failed ({e}). Continuing without authentication — this is expected if the target environment has no auth system."
                )
                self.token_manager = None

    def _get_url(self, endpoint):
        if self.secure:
            protocol = "https"
        else:
            protocol = "http"
        if self.port:
            return f"{protocol}://{self.host}:{self.port}/{endpoint}"
        return f"{protocol}://{self.host}/{endpoint}"

    def _get_url_api_v1(self, endpoint):
        return self._get_url(f"api/v1/{endpoint}")

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get headers with current access token.

        If TokenManager is configured, this will automatically refresh the token if needed.

        :param additional_headers: Optional additional headers to merge.
        :type additional_headers: dict
        :return: Headers dictionary with Authorization header.
        :rtype: dict
        """
        if self.access_token_provider:
            tok = self.access_token_provider()
            headers = {}
            if tok:
                headers["Authorization"] = f"{self.auth_scheme} {tok}"
        elif self.token_manager:
            # Get fresh token from TokenManager (will auto-refresh if needed)
            headers = self.token_manager.get_auth_header()
        else:
            # Use static headers
            headers = self.headers.copy()

        # Merge additional headers if provided
        if additional_headers:
            headers.update(additional_headers)

        return headers

    def _perform_chunked_upload(self, file_path: str) -> str:
        """Uploads a file in chunks and returns a file_token."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Upload failed: File not found ({file_path})")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        current_chunk_size = 900 * 1024  # Default chunk size to start from
        min_chunk_size = 256 * 1024  # Minimum chunk size when to give up

        while True:
            try:
                return self._do_chunked_upload(file_path, file_name, file_size, current_chunk_size)
            except requests.exceptions.HTTPError as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 413:
                    next_size = current_chunk_size // 2
                    if next_size < min_chunk_size:
                        raise RuntimeError(f"Upload failed: proxy rejects all supported chunk sizes (minimum {min_chunk_size // 1024} KB).")
                    ScaleoutLogger().warning(f"Proxy rejected chunk size (413). Retrying with {next_size // 1024} KB chunks...")
                    current_chunk_size = next_size
                    continue
                raise

    def _do_chunked_upload(self, file_path: str, file_name: str, file_size: int, chunk_size: int) -> str:
        headers = self._get_headers()

        # 1. Initialize Upload
        init_url = self._get_url_api_v1("file-upload/init")
        init_response = requests.post(
            init_url,
            json={"file_name": file_name, "file_size": file_size, "chunk_size": chunk_size},
            headers=headers,
            verify=self.verify,
        )
        init_response.raise_for_status()

        upload_id = init_response.json().get("upload_id")
        if not upload_id:
            raise RuntimeError("Failed to receive an upload_id from the backend")

        # 2. Upload Chunks
        chunk_url = self._get_url_api_v1(f"file-upload/{upload_id}/chunk")

        with open(file_path, "rb") as f:
            chunk_index = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break

                chunk_headers = headers.copy()
                chunk_headers["X-Chunk-Index"] = str(chunk_index)

                try:
                    chunk_resp = requests.post(chunk_url, data=chunk_data, headers=chunk_headers, verify=self.verify)
                    chunk_resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    if getattr(e, "response", None) is not None and e.response.status_code == 413:
                        abort_url = self._get_url_api_v1(f"file-upload/{upload_id}/abort")
                        requests.post(abort_url, headers=headers, verify=self.verify)
                    raise

                chunk_index += 1

        # 3. Complete Upload
        complete_url = self._get_url_api_v1(f"file-upload/{upload_id}/complete")
        complete_resp = requests.post(complete_url, headers=headers, verify=self.verify)
        complete_resp.raise_for_status()

        json_resp = complete_resp.json()
        file_token = json_resp.get("file_token")
        if not file_token:
            raise RuntimeError("Failed to fetch file_token. Backend completed upload but emitted no token.")

        return file_token

    # --- Clients --- #

    def get_client(self, id: str):
        """Get a client from the statestore.

        :param id: The client id to get.
        :type id: str
        :return: Client.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"clients/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_clients(self, n_max: int = None):
        """Get clients from the statestore.

        :param n_max: The maximum number of clients to get (If none all will be fetched).
        :type n_max: int
        :return: Clients.
        :rtype: dict
        """
        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("clients/"), verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_clients_count(self, only_active: bool = False):
        """Get the number of clients in the statestore.

        :return: The number of clients.
        :rtype: dict
        """
        _params = {}
        if only_active:
            _params["status"] = "online"
        response = requests.get(self._get_url_api_v1("clients/count"), params=_params, verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_client_config(self, checksum=True):
        """Get client config from controller. Optionally include the checksum.
        The config is used for clients to connect to the controller and ask for combiner assignment.

        :param checksum: Whether to include the checksum of the package.
        :type checksum: bool
        :return: The client configuration.
        :rtype: dict
        """
        _params = {"checksum": "true" if checksum else "false"}

        response = requests.get(self._get_url_api_v1("clients/config"), params=_params, verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_active_clients(self, combiner_id: str = None, n_max: int = None):
        """Get active clients from the statestore.

        :param combiner_id: The combiner id to get active clients for.
        :type combiner_id: str
        :param n_max: The maximum number of clients to get (If none all will be fetched).
        :type n_max: int
        :return: Active clients.
        :rtype: dict
        """
        _params = {"status": "online"}

        if combiner_id:
            _params["combiner"] = combiner_id

        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("clients/"), params=_params, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    # --- Combiners --- #

    def get_combiner(self, id: str):
        """Get a combiner from the statestore.

        :param id: The combiner id to get.
        :type id: str
        :return: Combiner.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"combiners/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_combiners(self, n_max: int = None):
        """Get combiners in the network.

        :param n_max: The maximum number of combiners to get (If none all will be fetched).
        :type n_max: int
        :return: Combiners.
        :rtype: dict
        """
        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("combiners/"), verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_combiners_count(self):
        """Get the number of combiners in the statestore.

        :return: The number of combiners.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("combiners/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    # --- Controllers --- #

    def get_controller_status(self):
        """Get the status of the controller.

        :return: The status of the controller.
        :rtype: dict
        """
        response = requests.get(self._get_url("get_controller_status"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    # --- Models --- #

    def get_model(self, id: str):
        """Get a model from the statestore.

        :param id: The id (or model property) of the model to get.
        :type id: str
        :return: Model.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"models/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_models(self, session_id: str = None, n_max: int = None):
        """Get models from the statestore.

        :param session_id: The session id to get models for. (optional)
        :type session_id: str
        :param n_max: The maximum number of models to get (If none all will be fetched).
        :type n_max: int
        :return: Models.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("models/"), params=_params, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_models_count(self):
        """Get the number of models in the statestore.

        :return: The number of models.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("models/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_active_model(self):
        """Get the latest model from the statestore.

        :return: The latest model.
        :rtype: dict
        """
        additional_headers = {"X-Limit": "1", "X-Sort-Key": "committed_at", "X-Sort-Order": "desc"}

        response = requests.get(self._get_url_api_v1("models/"), verify=self.verify, headers=self._get_headers(additional_headers))
        _json = response.json()

        if "result" in _json and len(_json["result"]) > 0:
            return _json["result"][0]

        return _json

    def get_model_trail(self, id: str = None, include_self: bool = True, reverse: bool = True, n_max: int = None):
        """Get the model trail.

        :param id: The id (or model property) of the model to start the trail from. (optional)
        :type id: str
        :param n_max: The maximum number of models to get (If none all will be fetched).
        :type n_max: int
        :return: Models.
        :rtype: dict
        """
        if not id:
            model = self.get_active_model()
            if "model_id" in model:
                id = model["model_id"]
            else:
                return model

        _count: int = n_max if n_max else self.get_models_count()

        additional_headers = {"X-Limit": str(_count), "X-Reverse": "true" if reverse else "false"}

        _include_self_str: str = "true" if include_self else "false"

        response = requests.get(
            self._get_url_api_v1(f"models/{id}/ancestors?include_self={_include_self_str}"), verify=self.verify, headers=self._get_headers(additional_headers)
        )
        _json = response.json()

        return _json

    def download_model(self, id: str, path: str):
        """Download the model with id id.

        :param id: The id (or model property) of the model to download.
        :type id: str
        :param path: The path to download the model to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"models/{id}/download"), verify=self.verify, headers=self._get_headers())

        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            return {"success": True, "message": "Model downloaded successfully."}
        else:
            return {"success": False, "message": "Failed to download model."}

    def set_active_model(self, path):
        """Set the initial model in the statestore and upload to model repository.

        :param path: The file path of the initial model to set.
        :type path: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        if path.endswith(".npz"):
            helper = "numpyhelper"
        elif path.endswith(".bin"):
            helper = "binaryhelper"

        if helper:
            response = requests.put(self._get_url_api_v1("helpers/active"), json={"helper": helper}, verify=self.verify, headers=self._get_headers())

        file_token = self._perform_chunked_upload(path)

        response = requests.post(
            self._get_url_api_v1("models/"), data={"helper": helper, "file_token": file_token}, verify=self.verify, headers=self._get_headers()
        )
        return response.json()

    # --- Packages --- #

    def get_package(self, id: str):
        """Get a compute package from the statestore.

        :param id: The id of the compute package to get.
        :type id: str
        :return: Package.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"packages/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_packages(self, n_max: int = None):
        """Get compute packages from the statestore.

        :param n_max: The maximum number of packages to get (If none all will be fetched).
        :type n_max: int
        :return: Packages.
        :rtype: dict
        """
        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("packages/"), verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_packages_count(self):
        """Get the number of compute packages in the statestore.

        :return: The number of packages.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_active_package(self):
        """Get the (active) compute package from the statestore.

        :return: Package.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/active"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_package_checksum(self):
        """Get the checksum of the compute package.

        :return: The checksum.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/checksum"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def download_package(self, path: str):
        """Download the compute package.

        :param path: The path to download the compute package to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/download"), verify=self.verify, headers=self._get_headers())
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            return {"success": True, "message": "Package downloaded successfully."}
        else:
            return {"success": False, "message": "Failed to download package."}

    def set_active_package(self, path: str, helper: str, name: str, description: str = ""):
        """Set the compute package in the statestore.

        :param path: The file path of the compute package to set.
        :type path: str
        :param helper: The helper type to use.
        :type helper: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        file_token = self._perform_chunked_upload(path)

        response = requests.post(
            self._get_url_api_v1("packages/"),
            data={"helper": helper, "name": name, "description": description, "file_token": file_token, "file_name": os.path.basename(path)},
            verify=self.verify,
            headers=self._get_headers(),
        )

        _json = response.json()

        return _json

    # --- Rounds --- #

    def get_round(self, id: str):
        """Get a round from the statestore.

        :param round_id: The round id to get.
        :type round_id: str
        :return: Round (config and metrics).
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"rounds/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_rounds(self, n_max: int = None, filter: Dict = None):
        """Get all rounds from the statestore.

        :param n_max: The maximum number of rounds to get (If none all will be fetched).
        :type n_max: int
        :return: Rounds.
        :rtype: dict
        """
        additional_headers = {}
        params = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        if filter:
            params.update({k: str(v) for k, v in filter.items()})

        response = requests.get(self._get_url_api_v1("rounds/"), verify=self.verify, headers=self._get_headers(additional_headers), params=params)

        _json = response.json()

        return _json

    def get_rounds_count(self):
        """Get the number of rounds in the statestore.

        :return: The number of rounds.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("rounds/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    # --- Sessions --- #

    def get_session(self, id: str = None, name: str = None):
        """Get a session from the statestore.

        :param id: The session id to get.
        :type id: str
        :param name: The session name to get.
        :type name: str
        :return: Session.
        :rtype: dict
        """
        if name:
            response = requests.get(self._get_url_api_v1(f"sessions?name={name}"), verify=self.verify, headers=self._get_headers())
            _json = response.json()
            if "result" in _json and len(_json["result"]) > 0:
                _json = _json["result"][0]
            else:
                _json = {"message": "Session not found."}
        elif id:
            response = requests.get(self._get_url_api_v1(f"sessions/{id}"), verify=self.verify, headers=self._get_headers())
            _json = response.json()
        else:
            _json = {"message": "No id or name provided."}

        return _json

    def get_sessions(self, n_max: int = None, name: str = None):
        """Get sessions from the statestore.

        :param n_max: The maximum number of sessions to get (If none all will be fetched).
        :type n_max: int
        :param name: The session name to get.
        :type name: str
        :return: Sessions.
        :rtype: dict
        """
        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        url = self._get_url_api_v1("sessions/")
        if name:
            url += f"?name={name}"

        response = requests.get(url, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_sessions_count(self):
        """Get the number of sessions in the statestore.

        :return: The number of sessions.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("sessions/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_session_status(self, id: str):
        """Get the status of a session.

        :param id: The id of the session to get.
        :type id: str
        :return: The status of the session.
        :rtype: str
        """
        session = self.get_session(id)

        if session and "status" in session:
            return session["status"]

        return "Could not retrieve session status."

    def session_is_finished(self, id: str):
        """Check if a session with id has finished.

        :param id: The id of the session to get.
        :type id: str
        :return: True if session is finished, otherwise false.
        :rtype: bool
        """
        status = self.get_session_status(id)
        return status and status.lower() == "finished"

    def run_custom_command(self, command_type: str, blocking: bool = False, timeout: int = None, client_ids: list = None, parameters: dict = None):
        if not command_type.startswith("Custom_"):
            command_type = "Custom_" + command_type

        url = self._get_url_api_v1("control/run_command")
        data = {"command_type": command_type, "blocking": blocking, "timeout": timeout, "client_ids": client_ids, **(parameters or {})}
        response = requests.post(url, json=data, verify=self.verify, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": f"Failed to run command. Status code: {response.status_code}, Response: {response.text}"}

    def start_session(
        self,
        name: str = None,
        aggregator: str = "fedavg",
        aggregator_kwargs: dict = None,
        model_id: str = None,
        round_timeout: int = 180,
        rounds: int = 5,
        round_buffer_size: int = -1,
        delete_models: bool = True,
        validate: bool = True,
        helper: str = None,
        min_clients: int = 1,
        requested_clients: int = 8,
        server_functions: ServerFunctionsBase = None,
    ):
        """Start a new session.

        :param name: The name of the session
        :type name: str
        :param aggregator: The aggregator plugin to use.
        :type aggregator: str
        :param model_id: The id of the initial model.
        :type model_id: str
        :param round_timeout: The round timeout to use in seconds.
        :type round_timeout: int
        :param rounds: The number of rounds to perform.
        :type rounds: int
        :param round_buffer_size: The round buffer size to use.
        :type round_buffer_size: int
        :param delete_models: Whether to delete models after each round at combiner (save storage).
        :type delete_models: bool
        :param validate: Whether to validate the model after each round.
        :type validate: bool
        :param helper: The helper type to use.
        :type helper: str
        :param min_clients: The minimum number of clients required.
        :type min_clients: int
        :param requested_clients: The requested number of clients.
        :type requested_clients: int
        :return: A dict with success or failure message and session config.
        :rtype: dict
        """
        if model_id is None:
            additional_headers = {"X-Limit": "1", "X-Sort-Key": "committed_at", "X-Sort-Order": "desc"}
            response = requests.get(self._get_url_api_v1("models/"), verify=self.verify, headers=self._get_headers(additional_headers))
            if response.status_code == 200:
                json = response.json()
                if "result" in json and len(json["result"]) > 0:
                    model_id = json["result"][0]["model_id"]
                else:
                    return {"message": "No models found in the repository"}
            else:
                return {"message": "No models found in the repository"}

        response = requests.post(
            self._get_url_api_v1("sessions/"),
            json={
                "name": name,
                "seed_model_id": model_id,
                "session_config": {
                    "aggregator": aggregator,
                    "aggregator_kwargs": aggregator_kwargs,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                    "buffer_size": round_buffer_size,
                    "model_id": model_id,
                    "delete_models_storage": delete_models,
                    "clients_required": min_clients,
                    "requested_clients": requested_clients,
                    "validate": validate,
                    "helper_type": helper,
                    "server_functions": None if server_functions is None else inspect.getsource(server_functions),
                },
            },
            verify=self.verify,
            headers=self._get_headers(),
        )

        if response.status_code == 201:
            session_id = response.json()["session_id"]
            response = requests.post(
                self._get_url_api_v1("sessions/start"),
                json={
                    "session_id": session_id,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                },
                verify=self.verify,
                headers=self._get_headers(),
            )
            # Try to parse JSON, but handle the case where it fails
            try:
                response_json = response.json()
                response_json["session_id"] = session_id
                return response_json
            except requests.exceptions.JSONDecodeError:
                # Handle invalid JSON response
                return {"success": response.status_code < 400, "session_id": session_id, "message": f"Session started with status code {response.status_code}"}

        _json = response.json()
        return _json

    def continue_session(self, session_id: str, rounds: int = 5, round_timeout: int = 180):
        """Continue a session.

        :param session_id: The id of the session to continue.
        :type session_id: str
        :param rounds: The number of rounds to perform.
        :type rounds: int
        :param round_timeout: The round timeout to use in seconds.
        :type round_timeout: int
        :return: A dict with success or failure message and session config.
        :rtype: dict
        """
        if not session_id:
            return {"message": "No session id provided."}
        if rounds is None or rounds <= 0:
            return {"message": "Invalid number of rounds provided. Must be greater than 0."}
        if round_timeout is None or round_timeout <= 0:
            return {"message": "Invalid round timeout provided. Must be greater than 0."}
        # Check if session exists
        session = self.get_session(session_id)
        if not session or "session_id" not in session:
            return {"message": "Session not found."}
        # Check if session is finished
        if not self.session_is_finished(session_id):
            return {"message": "Session is already running."}

        response = requests.post(
            self._get_url_api_v1("sessions/start"),
            json={
                "session_id": session_id,
                "rounds": rounds,
                "round_timeout": round_timeout,
            },
            verify=self.verify,
            headers=self._get_headers(),
        )

        _json = response.json()

        return _json

    # --- Statuses --- #

    def get_status(self, id: str):
        """Get a status object (event) from the statestore.

        :param id: The id of the status to get.
        :type id: str
        :return: Status.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"statuses/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_statuses(self, session_id: str = None, event_type: str = None, sender_name: str = None, sender_role: str = None, n_max: int = None):
        """Get statuses from the statestore. Filter by input parameters

        :param session_id: The session id to get statuses for.
        :type session_id: str
        :param event_type: The event type to get.
        :type event_type: str
        :param sender_name: The sender name to get.
        :type sender_name: str
        :param sender_role: The sender role to get.
        :type sender_role: str
        :param n_max: The maximum number of statuses to get (If none all will be fetched).
        :type n_max: int
        :return: Statuses
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        if event_type:
            _params["type"] = event_type

        if sender_name:
            _params["sender.name"] = sender_name

        if sender_role:
            _params["sender.role"] = sender_role

        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("statuses/"), params=_params, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_statuses_count(self):
        """Get the number of statuses in the statestore.

        :return: The number of statuses.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("statuses/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    # --- Validations --- #

    def get_validation(self, id: str):
        """Get a validation from the statestore.

        :param id: The id of the validation to get.
        :type id: str
        :return: Validation.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"validations/{id}"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    def get_validations(
        self,
        session_id: str = None,
        model_id: str = None,
        correlation_id: str = None,
        sender_name: str = None,
        sender_role: str = None,
        sender_client_id: str = None,
        n_max: int = None,
    ):
        """Get validations from the statestore. Filter by input parameters.

        :param session_id: The session id to get validations for.
        :type session_id: str
        :param model_id: The model id to get validations for.
        :type model_id: str
        :param correlation_id: The correlation id to get validations for.
        :type correlation_id: str
        :param sender_name: The sender name to get validations for.
        :type sender_name: str
        :param sender_role: The sender role to get validations for.
        :type sender_role: str
        :param sender_client_id: The sender client id to get validations for.
        :type sender_client_id: str
        :param n_max: The maximum number of validations to get (If none all will be fetched).
        :type n_max: int
        :return: Validations.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        if model_id:
            _params["model_id"] = model_id

        if correlation_id:
            _params["correlation_id"] = correlation_id

        if sender_name:
            _params["sender.name"] = sender_name

        if sender_role:
            _params["sender.role"] = sender_role

        if sender_client_id:
            _params["sender.client_id"] = sender_client_id

        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("validations/"), params=_params, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def get_validations_count(self, session_id: str = None):
        """Get the number of validations in the statestore.

        :return: The number of validations.
        :rtype: dict
        """
        _params = {}
        if session_id:
            _params["session_id"] = session_id
        response = requests.get(self._get_url_api_v1("validations/count"), params=_params, verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

        response = requests.get(self._get_url_api_v1("validations/count"), verify=self.verify, headers=self._get_headers())

        _json = response.json()

        return _json

    # --- Predictions --- #

    def get_predictions(
        self,
        model_id: str = None,
        correlation_id: str = None,
        client_id: str = None,
        n_max: int = None,
    ):
        """Get predictions from the statestore. Filter by input parameters.

        :param model_id: The model id to get predictions for.
        :type model_id: str
        :param correlation_id: The correlation id to get predictions for.
        :type correlation_id: str
        :param client_id: The client id to get predictions for.
        :type client_id: str
        :param n_max: The maximum number of predictions to get (If none all will be fetched).
        :type n_max: int
        :return: Predictions.
        :rtype: dict
        """
        _params = {}

        if model_id:
            _params["model_id"] = model_id

        if correlation_id:
            _params["correlation_id"] = correlation_id

        if client_id:
            _params["client_id"] = client_id

        additional_headers = {}
        if n_max:
            additional_headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("predictions/"), params=_params, verify=self.verify, headers=self._get_headers(additional_headers))

        _json = response.json()

        return _json

    def start_predictions(self, prediction_id: str = None, model_id: str = None):
        """Start predictions for a model.

        :param model_id: The model id to start predictions for.
        :type model_id: str
        :param data: The data to predict.
        :type data: dict
        :return: A dict with success or failure message.
        :rtype: dict
        """
        if not prediction_id:
            prediction_id = str(uuid.uuid4())
        response = requests.post(
            self._get_url_api_v1("predictions/start"),
            json={"prediction_id": prediction_id, "model_id": model_id},
            verify=self.verify,
            headers=self._get_headers(),
        )

        _json = response.json()

        return _json

    # --- Attributes --- #

    def get_current_attributes(self, node_list):
        """Get the current attributes of the node(s).

        :param node_list: The list of nodes to get the attributes for or a single node_id (client_id or combiner_id)
        :type node_list: list|str
        :raises ValueError: If node_list is not a list or empty.
        :return: The current attributes of the nodes.
        :rtype: dict
        """
        if not isinstance(node_list, list):
            if isinstance(node_list, str):
                node_list = [node_list]
            else:
                raise ValueError("node_list must be a list or string")
        if len(node_list) == 0:
            raise ValueError("node_list must not be empty")
        json = {"node_ids": node_list}
        response = requests.post(self._get_url_api_v1("attributes/current"), json=json, verify=self.verify, headers=self._get_headers())
        _json = response.json()
        return _json

    def add_attributes(self, attribute: dict) -> dict:
        """Add or update node attributes via the controller API.

        :param attribute: A dict matching AttributeDTO.schema, e.g.:

        .. code-block:: text

           {
               "key": "charging",
               "value": "true",
               "sender": {
                   "name": "",
                   "role": "",
                   "client_id": "abc123"    # or "combiner_id": "abc123"
               }
           }

        :return: Parsed JSON response from the server.
        :rtype: dict
        """
        url = self._get_url_api_v1("attributes/")
        response = requests.post(url, json=attribute, headers=self._get_headers(), verify=self.verify)
        response.raise_for_status()
        return response.json()

    def add_status(self, status: dict) -> dict:
        """Submit a status entry to the controller API.

        :param status: A dict matching StatusDTO.schema, e.g.:

            .. code-block:: json

               {
                   "type": "MODEL_UPDATE",
                   "log_level": "INFO",
                   "status": "Training complete",
                   "sender": {
                       "client_id": "abc-123"
                   }
               }

        :return: Parsed JSON response from the server.
        :rtype: dict
        """
        url = self._get_url_api_v1("statuses/")
        response = requests.post(url, json=status, headers=self._get_headers(), verify=self.verify)
        response.raise_for_status()
        return response.json()

    def get_attribute_trail(self, node_id: str):
        """Get the full attribute history for a given node.

        :param node_id: The node_id (client or combiner) to fetch attributes for
        :type node_id: str
        :return: All matching attributes for the node
        :rtype: dict
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")

        # Reuse existing /attributes/list endpoint with a filter on node_id
        url = self._get_url_api_v1("attributes/list")
        response = requests.post(
            url,
            json={"node_id": node_id},
            headers=self._get_headers(),
            verify=self.verify,
        )
        response.raise_for_status()
        return response.json()

    ### Control Functions ###
    def step_current_session(self):
        """Continue a session control.

        :param session_id: The id of the session to continue.
        :type session_id: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        response = requests.post(
            self._get_url_api_v1("control/continue"),
            verify=self.verify,
            headers=self._get_headers(),
        )

        _json = response.json()

        return _json

    def stop_current_session(self):
        """Stop a session control.

        :param session_id: The id of the session to stop.
        :type session_id: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        response = requests.post(
            self._get_url_api_v1("control/stop"),
            verify=self.verify,
            headers=self._get_headers(),
        )

        _json = response.json()

        return _json
