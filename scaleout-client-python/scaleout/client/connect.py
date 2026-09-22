"""Connector class for assigning clients to the Scaleout network via the discovery service (REST-API).

Once assigned, the client will retrieve combiner assignment from the discovery service.
The discovery service will also add the client to the statestore.
"""

import enum
from typing import Dict, Optional, Tuple
import uuid

import requests

from scaleoututil.config import (
    SCALEOUT_AUTH_REFRESH_TOKEN,
    SCALEOUT_AUTH_SCHEME,
    SCALEOUT_CUSTOM_URL_PREFIX,
)
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.http_status_codes import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_NO_CONTENT, HTTP_STATUS_OK, HTTP_STATUS_UNAUTHORIZED
from scaleoututil.auth.login import Login

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


def get_url(api_url: str) -> str:
    """Construct the URL for the API."""
    return f"{api_url}/{SCALEOUT_CUSTOM_URL_PREFIX}".rstrip("/")


class ClientOptions:
    """Options for configuring the client."""

    def __init__(self, name: str, package: str, preferred_combiner: Optional[str] = None, client_id: Optional[str] = None) -> None:
        """Initialize ClientOptions with validation."""
        self._validate(name, package)
        self.name = name
        self.package = package
        self.preferred_combiner = preferred_combiner
        self.client_id = client_id if client_id else str(uuid.uuid4())

    def _validate(self, name: str, package: str) -> None:
        """Validate the name and package."""
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Name must be a string")
        if not isinstance(package, str) or len(package) == 0 or package not in ["local", "remote"]:
            raise ValueError("Package must be either 'local' or 'remote'")

    def to_json(self) -> Dict[str, Optional[str]]:
        """Convert ClientOptions to JSON."""
        return {
            "name": self.name,
            "client_id": self.client_id,
            "preferred_combiner": self.preferred_combiner,
            "package": self.package,
        }


class Status(enum.Enum):
    """Enum for representing the status of a client assignment."""

    Unassigned = 0
    Assigned = 1
    TryAgain = 2
    UnAuthorized = 3
    UnMatchedConfig = 4


class ConnectorClient:
    """Connector for assigning client to a combiner in the Scaleout network.

    :param host: host of discovery service
    :type host: str
    :param port: port of discovery service
    :type port: int
    :param token: token for authentication
    :type token: str
    :param name: name of client
    :type name: str
    :param remote_package: True if remote package is used, False if local
    :type remote_package: bool
    :param force_ssl: True if https is used, False if http
    :type force_ssl: bool
    :param verify: True if certificate is verified, False if not
    :type verify: bool
    :param combiner: name of preferred combiner
    :type combiner: Optional[str]
    :param id: id of client
    :type id: Optional[str]
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str,
        name: str,
        remote_package: bool,
        force_ssl: bool = False,
        verify: bool = True,
        combiner: Optional[str] = None,
        id: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_endpoint: Optional[str] = None,
    ) -> None:
        """Initialize the ConnectorClient.

        Args:
            host: Host of discovery service
            port: Port of discovery service
            token: Initial access token for authentication
            name: Name of client
            remote_package: True if remote package is used, False if local
            force_ssl: True if https is used, False if http
            verify: True if certificate is verified (default: True), False for development with self-signed certificates
            combiner: Name of preferred combiner
            id: ID of client
            refresh_token: Refresh token for automatic token renewal
            token_endpoint: Custom token endpoint URL (overrides env var)

        """
        self.host = host
        self.port = port
        self.name = name
        self.verify = verify
        self.preferred_combiner = combiner
        self.id = id
        self.package = "remote" if remote_package else "local"

        # for https we assume an ingress handles permanent redirect (308)
        self.prefix = "https://" if force_ssl else "http://"
        self.connect_string = f"{self.prefix}{self.host}:{self.port}" if self.port else f"{self.prefix}{self.host}"

        # Initialize Login with refresh capability
        credential = refresh_token or SCALEOUT_AUTH_REFRESH_TOKEN
        if credential:
            self._login = Login(
                server_url=self.connect_string,
                credential=credential,
                verify_ssl=self.verify,
                client_id=self.id,
            )
            self.token_manager = None  # kept for backward compatibility
            ScaleoutLogger().debug("Login initialized with refresh token support")
        else:
            self._login = None
            self.token_manager = None
            self.token = token
            ScaleoutLogger().warning("No refresh token provided - automatic token refresh disabled")

        ScaleoutLogger().debug(f"Setting connection string to {self.connect_string}.")

    def _get_current_token(self) -> str:
        """Get current access token, using TokenManager if available.

        Returns:
            Current access token

        """
        if self._login:
            return self._login.get_access_token()
        return self.token

    def assign(self) -> Tuple[Status, Optional[dict]]:
        """Connect client to Scaleout network discovery service, ask for combiner assignment.

        :return: Tuple with assignment status, combiner connection information if successful, else None.
        :rtype: tuple(:class:`scaleout.network.clients.connect.Status`, Optional[dict])
        """
        try:
            payload = {
                "name": self.name,
                "client_id": self.id,
                "preferred_combiner": self.preferred_combiner,
                "package": self.package,
            }

            # Get current token (will auto-refresh if needed)
            current_token = self._get_current_token()

            retval = requests.post(
                self.connect_string + SCALEOUT_CUSTOM_URL_PREFIX + "/add_client",
                json=payload,
                verify=self.verify,
                allow_redirects=True,
                headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {current_token}"},
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            ScaleoutLogger().debug(f"***** {e}")
            return Status.Unassigned, {}

        if retval.status_code == HTTP_STATUS_BAD_REQUEST:
            reason = retval.json()["message"]
            return Status.UnMatchedConfig, reason

        if retval.status_code == HTTP_STATUS_UNAUTHORIZED:
            reason = retval.json().get("message", "Unauthorized connection to reducer, make sure the correct token is set")
            ScaleoutLogger().warning(reason)

            # On a server-reported token expiry, force a refresh via Login (the local
            # token can still look valid here due to clock skew) and signal a retry.
            if self._login and reason == "Token expired":
                try:
                    ScaleoutLogger().info("Access token expired; forcing refresh...")
                    if self._login.refresh() is not None:
                        ScaleoutLogger().info("Token refreshed, retrying assignment...")
                        return Status.TryAgain, reason
                except Exception as e:
                    ScaleoutLogger().error(f"Token refresh failed: {e}")
                    return Status.UnAuthorized, "Could not refresh token"

            return Status.UnAuthorized, reason

        if HTTP_STATUS_OK <= retval.status_code < HTTP_STATUS_NO_CONTENT:
            if retval.json().get("status") == "retry":
                reason = retval.json().get("message", "Controller was not ready. Try again later.")
                return Status.TryAgain, reason

            return Status.Assigned, retval.json()

        return Status.Unassigned, None
