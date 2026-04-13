"""Token manager for handling access and refresh tokens in Scaleout clients."""

import jwt
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
import requests

from scaleoututil.config import SCALEOUT_AUTH_REFRESH_TOKEN_URI, SCALEOUT_AUTH_SCHEME
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.http_status_codes import HTTP_STATUS_OK, HTTP_STATUS_NO_CONTENT, HTTP_STATUS_UNAUTHORIZED

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds

# Buffer time before token expires to trigger refresh (30 seconds)
# This should be less than half the token lifetime to avoid immediate refresh loops
TOKEN_REFRESH_BUFFER_SECONDS = 30


class TokenManager:
    """Manages access and refresh tokens with automatic refresh capabilities.

    This class provides thread-safe token management with automatic refresh
    when tokens are about to expire. It ensures that only one refresh operation
    happens at a time across all threads.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_endpoint: Optional[str] = None,
        expires_in: Optional[int] = None,
        verify_ssl: bool = True,
        role: str = "client",
        on_token_refresh: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> None:
        """Initialize the TokenManager.

        Args:
            access_token: Initial access token (optional if refresh_token is provided)
            refresh_token: Refresh token for obtaining new access tokens
            token_endpoint: URL endpoint for token refresh (overrides env var)
            expires_in: Token lifetime in seconds (optional, will be extracted from JWT if not provided)
            verify_ssl: Whether to verify SSL certificates (default: True). Set to False for development with self-signed certificates.
            role: Role associated with the token (default: "client")
            on_token_refresh: Optional callback function(access_token, refresh_token, expires_at) called when tokens are refreshed

        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self.role = role
        self._token_endpoint = SCALEOUT_AUTH_REFRESH_TOKEN_URI or token_endpoint
        self._verify_ssl = verify_ssl
        self._lock = threading.Lock()
        self._refresh_promise: Optional[threading.Event] = None
        self._on_token_refresh = on_token_refresh

        if not self._token_endpoint:
            ScaleoutLogger().warning("No token endpoint provided; token refresh will not be available.")

        # If no access token provided but refresh token is available, perform initial refresh
        if not self._access_token and self._refresh_token:
            ScaleoutLogger().info("No access token provided, performing initial token exchange...")
            try:
                self._perform_token_refresh()
                ScaleoutLogger().info(f"Initial token obtained. Token expires at: {self._token_expires_at}")
            except RuntimeError as e:
                ScaleoutLogger().error(f"Initial token exchange failed: {e}")
                ScaleoutLogger().info("Hint: If you have a cached token, try running without the --token flag to use the cached refresh token.")
                raise
        elif not self.role:
            ScaleoutLogger().error("No role provided; token management will not function correctly.")
        else:
            # Extract expiration from JWT or use provided expires_in
            if self._access_token:
                self._token_expires_at = self._extract_expiration(self._access_token, expires_in)
            else:
                # No token yet, set a default
                expires_in = expires_in or 3600
                self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            ScaleoutLogger().info(f"TokenManager initialized. Token expires at: {self._token_expires_at}")

    def get_access_token(self) -> str:
        """Get current access token, refreshing if necessary.

        This method is thread-safe and will automatically refresh the token
        if it's close to expiring. If a refresh is already in progress, it
        will wait for that refresh to complete.

        Returns:
            Current valid access token

        Raises:
            RuntimeError: If token refresh fails

        """
        # Check if token needs refresh (with buffer time)

        if self._should_refresh_token():
            with self._lock:
                # Double-check after acquiring lock (another thread might have refreshed)
                if self._should_refresh_token():
                    # Check if a refresh is already in progress
                    if self._refresh_promise is not None:
                        # Wait for the ongoing refresh
                        ScaleoutLogger().debug("Token refresh already in progress, waiting...")
                        refresh_event = self._refresh_promise
                        # Release lock while waiting
                        self._lock.release()
                        try:
                            refresh_event.wait(timeout=30)
                        finally:
                            self._lock.acquire()
                    else:
                        # Start a new refresh
                        self._refresh_promise = threading.Event()
                        try:
                            self._perform_token_refresh()
                            self._refresh_promise.set()
                        except Exception as e:
                            self._refresh_promise.set()
                            self._refresh_promise = None
                            raise RuntimeError(f"Token refresh failed: {e}")
                        finally:
                            self._refresh_promise = None

        return self._access_token

    def _should_refresh_token(self) -> bool:
        """Check if token should be refreshed.

        Returns:
            True if token should be refreshed, False otherwise

        """
        if not self._refresh_token or not self._token_endpoint:
            return False

        # Refresh if token expires within the buffer time
        now = datetime.now(timezone.utc)
        time_until_expiry = (self._token_expires_at - now).total_seconds()
        should_refresh = time_until_expiry < TOKEN_REFRESH_BUFFER_SECONDS

        if should_refresh:
            ScaleoutLogger().debug(f"Token needs refresh. Expires in {time_until_expiry:.0f}s (buffer: {TOKEN_REFRESH_BUFFER_SECONDS}s)")

        return should_refresh

    def _extract_expiration(self, token: str, fallback_expires_in: Optional[int] = None) -> datetime:
        """Extract expiration time from JWT token.

        Args:
            token: JWT access token
            fallback_expires_in: Fallback expiration time in seconds if JWT decode fails

        Returns:
            Expiration datetime in UTC

        """
        try:
            # Decode without verification (we trust tokens from our refresh endpoint)
            decoded = jwt.decode(token, options={"verify_signature": False})
            exp_timestamp = decoded.get("exp")
            if exp_timestamp:
                # JWT exp is always in UTC seconds since epoch
                expiration = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                now = datetime.now(timezone.utc)
                time_until_exp = (expiration - now).total_seconds()
                ScaleoutLogger().debug(f"Extracted expiration from JWT: {expiration} (in {time_until_exp:.0f} seconds)")
                return expiration
        except Exception as e:
            ScaleoutLogger().warning(f"Failed to decode JWT for expiration: {e}")

        # Fallback to provided expires_in or default
        fallback = fallback_expires_in or 3600
        expiration = datetime.now(timezone.utc) + timedelta(seconds=fallback)
        ScaleoutLogger().debug(f"Using fallback expiration: {fallback} seconds from now ({expiration})")
        return expiration

    def _perform_token_refresh(self) -> None:
        """Perform the actual token refresh operation.

        This method should only be called while holding the lock.

        Raises:
            RuntimeError: If refresh fails

        """
        if not self._refresh_token:
            ScaleoutLogger().error("No refresh token available, cannot refresh access token.")
            return

        if not self._token_endpoint:
            ScaleoutLogger().error("No token endpoint configured, cannot refresh access token.")
            return

        ScaleoutLogger().debug("Refreshing access token...")

        try:
            response = requests.post(
                self._token_endpoint,
                json={"refresh_token": self._refresh_token, "role": self.role},
                verify=self._verify_ssl,
                allow_redirects=True,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == HTTP_STATUS_UNAUTHORIZED:
                error_text = response.text
                ScaleoutLogger().error(f"Refresh token is invalid or expired: {error_text}")
                ScaleoutLogger().debug("The refresh token may have been revoked or expired. You may need to log in again to obtain a new refresh token.")
                raise RuntimeError("Refresh token is invalid or expired. Please log in again to obtain a new token.")

            if not (HTTP_STATUS_OK <= response.status_code < HTTP_STATUS_NO_CONTENT):
                error_text = response.text

                # Detect common misconfiguration: HTTP request to HTTPS port
                if response.status_code == 400 and "plain HTTP request was sent to HTTPS port" in error_text:
                    ScaleoutLogger().error(
                        "Token refresh failed: HTTP request sent to HTTPS port."
                        "If using port 443 or an HTTPS endpoint, set secure=True in Scaleout initialization."
                    )
                    raise RuntimeError(
                        "Token refresh failed: HTTP request sent to HTTPS port. Please set secure=True when initializing Scaleout for HTTPS endpoints."
                    )

                ScaleoutLogger().error(f"Token refresh failed with status {response.status_code}: {error_text}")
                raise RuntimeError(f"Token refresh failed with status {response.status_code}")

            response_data = response.json()
            new_access_token = response_data.get("access_token") or response_data.get("access")
            new_refresh_token = response_data.get("refresh_token") or response_data.get("refresh")
            expires_in = response_data.get("expires_in")

            if not new_access_token:
                raise RuntimeError("No access token in refresh response")

            # Update tokens
            self._access_token = new_access_token
            if new_refresh_token:
                self._refresh_token = new_refresh_token

            # Extract expiration from JWT (with fallback to expires_in from response)
            self._token_expires_at = self._extract_expiration(new_access_token, expires_in)

            ScaleoutLogger().debug(f"Access token refreshed successfully. New expiration: {self._token_expires_at}")

            # Call the callback if provided (after setting expiration)
            if self._on_token_refresh:
                try:
                    self._on_token_refresh(self._access_token, self._refresh_token, self._token_expires_at)
                    ScaleoutLogger().debug("Token refresh callback executed successfully")
                except Exception as e:
                    ScaleoutLogger().error(f"Error in token refresh callback: {e}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error during token refresh: {e}")

    def update_tokens(self, access_token: str, refresh_token: Optional[str] = None, expires_in: Optional[int] = None) -> None:
        """Manually update tokens.

        This method allows updating tokens when they're obtained through
        other means (e.g., initial login).

        Args:
            access_token: New access token
            refresh_token: New refresh token (optional)
            expires_in: Token lifetime in seconds (optional)

        """
        with self._lock:
            self._access_token = access_token
            if refresh_token:
                self._refresh_token = refresh_token

            # Extract expiration from JWT or use provided expires_in
            self._token_expires_at = self._extract_expiration(access_token, expires_in)

            ScaleoutLogger().debug("Tokens updated manually")

    def get_auth_header(self) -> dict:
        """Get authorization header with current access token.

        Returns:
            Dictionary with Authorization header

        """
        token = self.get_access_token()
        return {"Authorization": f"{SCALEOUT_AUTH_SCHEME} {token}"}

    def is_token_expired(self) -> bool:
        """Check if the current token is expired.

        Returns:
            True if token is expired, False otherwise

        """
        return datetime.now(timezone.utc) >= self._token_expires_at
