"""Credential-aware authentication for Scaleout clients."""

import base64
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import jwt as _pyjwt
import requests

from scaleoututil.auth.token_cache import TokenCache
from scaleoututil.auth.token_manager import TokenManager
from scaleoututil.config import SCALEOUT_AUTH_SCHEME
from scaleoututil.logging import ScaleoutLogger


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without signature verification."""
    try:
        # Use PyJWT as the primary decoder — it handles all standard JWT formats
        # and is more robust than manual base64 parsing.
        # verify_exp=False so this works for type-detection even on expired tokens.
        return _pyjwt.decode(token, options={"verify_signature": False, "verify_exp": False})
    except Exception:
        pass
    # Fallback: manual base64url parsing for non-standard tokens
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return {}


def _expiry_from_jwt(token: str) -> datetime:
    """Extract expiration datetime from a JWT exp claim (UTC). Falls back to 1 hour from now."""
    claims = _decode_jwt_payload(token)
    exp = claims.get("exp")
    if exp:
        return datetime.fromtimestamp(exp, tz=timezone.utc)
    return datetime.now(tz=timezone.utc) + timedelta(hours=1)


def _enroll_client(
    server_url: str,
    enrollment_token: str,
    client_name: Optional[str] = None,
    client_id: Optional[str] = None,
    verify_ssl: bool = True,
) -> tuple:
    """Call POST /api/v1/clients/enroll with the enrollment token.

    Returns (client_id, access_token, refresh_token).
    Raises RuntimeError on auth failure or unexpected response.
    """
    url = server_url.rstrip("/") + "/api/v1/clients/enroll"
    body = {}
    if client_name:
        body["name"] = client_name
    if client_id:
        body["client_id"] = client_id
    resp = requests.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {enrollment_token}"},
        verify=verify_ssl,
        allow_redirects=True,
        timeout=10,
    )
    if resp.status_code == 401:
        raise RuntimeError("Enrollment token is invalid or expired")
    resp.raise_for_status()
    data = resp.json()
    return data["client_id"], data["access_token"], data["refresh_token"]


def _detect_credential_type(credential: str) -> str:
    """Detect credential type from JWT claims."""
    claims = _decode_jwt_payload(credential)
    if claims.get("token_type") == "api_key":
        return "api_key"
    role = claims.get("role", "")
    if role == "enrollment":
        return "enrollment"
    if role == "client_refresh":
        return "client_refresh"
    return "refresh_token"


def _safe_cache_id(server_url: str) -> str:
    return "login-" + re.sub(r"[^\w-]", "_", server_url)


class Login:
    """Credential-aware authentication for Scaleout clients.

    Accepts any Scaleout credential (refresh token, API key, enrollment token),
    detects the type from JWT claims, and runs the appropriate auth flow.
    Applications only call get_access_token() or get_auth_header().

    Credential type detection (from JWT claims, no signature verification):
        token_type=api_key   -> returned as-is, never refreshed
        role=enrollment      -> enrollment flow: calls /api/v1/clients/enroll, then uses client_refresh flow
        role=client_refresh  -> token refresh at /api/v1/clients/token
        anything else        -> token refresh at /api/auth/refresh
    """

    def __init__(
        self,
        server_url: str,
        credential: Optional[str] = None,
        verify_ssl: bool = True,
        client_id: Optional[str] = None,
        on_token_refresh: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> None:
        self._api_key: Optional[str] = None
        self._manager: Optional[TokenManager] = None
        self._enrolled_client_id: Optional[str] = None
        self._verify_ssl = verify_ssl
        self._ctype: Optional[str] = None
        # Optional caller callback (access_token, refresh_token, expires_at), invoked in addition
        # to the on-disk token cache whenever the tokens are refreshed. Used e.g. by the CLI to
        # persist a rotated refresh token back into its context file.
        self._on_token_refresh = on_token_refresh

        server_url = server_url.rstrip("/")

        if credential is None:
            # No credential supplied: fall back to cached client tokens for the given
            # client_id, if any (e.g. re-running a client with just --client-id after a
            # previous enrollment). With no credential and no client_id there is nothing to do.
            if client_id:
                self._init_from_cached_client_tokens(server_url, client_id)
            return

        self._ctype = _detect_credential_type(credential)

        if self._ctype == "api_key":
            self._api_key = credential
            return

        if self._ctype == "enrollment":
            # The enrollment token is authoritative: enroll with it. Pass the caller-supplied
            # client_id so the server reuses the same client on re-enrollment (idempotent)
            # rather than minting a new client record each time. Cache the issued tokens keyed
            # by the resulting client_id for the no-credential re-run path above.
            try:
                enrolled_id, access_token, refresh_token = _enroll_client(server_url, credential, client_id=client_id, verify_ssl=verify_ssl)
            except RuntimeError as e:
                ScaleoutLogger().warning(f"Enrollment failed ({e}). Continuing without authentication.")
                return
            self._enrolled_client_id = enrolled_id
            cache = TokenCache(cache_id=enrolled_id)
            cache.save(access_token, refresh_token, _expiry_from_jwt(access_token))
            self._manager = self._build_manager(f"{server_url}/api/v1/clients/token", access_token, refresh_token, cache)
            return

        endpoint = f"{server_url}/api/v1/clients/token" if self._ctype == "client_refresh" else f"{server_url}/api/auth/refresh"

        # For client_refresh tokens, key the cache by the enrolled client_id (JWT sub) so
        # different enrolled clients on the same server don't share a cached access token.
        if self._ctype == "client_refresh":
            sub = _decode_jwt_payload(credential).get("sub")
            cache_id = sub if sub else _safe_cache_id(server_url)
        else:
            cache_id = _safe_cache_id(server_url)
        cache = TokenCache(cache_id=cache_id)
        cached_access = cache.get_access_token() if cache.is_access_token_valid() else None
        self._manager = self._build_manager(endpoint, cached_access, credential, cache)

    def _init_from_cached_client_tokens(self, server_url: str, client_id: str) -> None:
        """Use previously cached client tokens for client_id (no enrollment), if present."""
        cache = TokenCache(cache_id=client_id)
        refresh_token = cache.get_refresh_token()
        if not refresh_token:
            ScaleoutLogger().warning(f"No cached credentials for client_id '{client_id}'. Continuing without authentication.")
            return
        self._enrolled_client_id = client_id
        cached_access = cache.get_access_token() if cache.is_access_token_valid() else None
        self._manager = self._build_manager(f"{server_url}/api/v1/clients/token", cached_access, refresh_token, cache)

    def _build_manager(self, endpoint: str, access_token: Optional[str], refresh_token: str, cache: TokenCache) -> Optional[TokenManager]:
        """Construct a TokenManager, returning None (and warning) if initialisation fails."""

        def _on_refresh(new_access: str, new_refresh: str, expires_at: datetime) -> None:
            cache.save(new_access, new_refresh, expires_at)
            if self._on_token_refresh is not None:
                self._on_token_refresh(new_access, new_refresh, expires_at)

        try:
            return TokenManager(
                access_token=access_token,
                refresh_token=refresh_token,
                token_endpoint=endpoint,
                verify_ssl=self._verify_ssl,
                on_token_refresh=_on_refresh,
            )
        except RuntimeError as e:
            ScaleoutLogger().warning(f"Token initialisation failed ({e}). Continuing without authentication.")
            return None

    @property
    def enrolled_client_id(self) -> Optional[str]:
        """Client ID assigned by the server during enrollment (None for non-enrollment flows)."""
        return self._enrolled_client_id

    @property
    def ctype(self) -> Optional[str]:
        """Credential type for the login session.

        api_key         -> returned as-is, never refreshed
        enrollment      -> enrollment flow: calls /api/v1/clients/enroll, then uses client_refresh flow
        client_refresh  -> token refresh at /api/v1/clients/token
        anything else   -> token refresh at /api/auth/refresh
        """
        return self._ctype

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing automatically if needed."""
        if self._api_key is not None:
            return self._api_key
        if self._manager is not None:
            return self._manager.get_access_token()
        raise RuntimeError("No credential configured")

    def get_auth_header(self) -> dict:
        """Return an Authorization header dict."""
        return {"Authorization": f"{SCALEOUT_AUTH_SCHEME} {self.get_access_token()}"}

    def refresh(self) -> Optional[str]:
        """Force a token refresh and return the new access token.

        Unlike :meth:`get_access_token`, this bypasses the local expiry check — use it
        when the server reports the current access token as expired even though the
        client's local view still considers it valid (e.g. clock skew). Returns None for
        credentials that cannot be refreshed (e.g. a static API key); raises if the
        refresh request itself fails.
        """
        if self._manager is None:
            return None
        with self._manager._lock:
            self._manager._perform_token_refresh()
        return self._manager.get_access_token()
