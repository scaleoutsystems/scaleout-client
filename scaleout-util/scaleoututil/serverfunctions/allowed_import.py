import os
import random  # noqa: F401
from typing import Any, Callable, Optional, Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from scaleoututil.api.client import Scaleout
from scaleoututil.logging import ScaleoutLogger  # noqa: F401
from scaleoututil.serverfunctions.serverfunctionsbase import ServerFunctionsBase, RoundType  # noqa: F401


import threading

_api_client_instance: Optional[Scaleout] = None
_api_client_lock = threading.RLock()

_api_host: Optional[str] = None
_api_port: Optional[int] = None
_api_token_function: Optional[Callable[[], Optional[str]]] = None


def configure_api_client(*, host: str, port: int, token_function: Optional[Callable[[], Optional[str]]] = None) -> None:
    """Configure globals backing `api_client`."""
    global _api_host, _api_port, _api_token_function, _api_client_instance
    with _api_client_lock:
        _api_host = host
        _api_port = int(port)
        if token_function is not None:
            _api_token_function = token_function
        _api_client_instance = None


def _default_host_port() -> tuple[str, int]:
    if _api_host is not None and _api_port is not None:
        return _api_host, _api_port

    env_host = os.getenv("SCALEOUT_API_HOST")
    env_port = os.getenv("SCALEOUT_API_PORT")
    if env_host and env_port:
        return env_host, int(env_port)

    return "scaleout-api-server", 8092


def _get_api_client() -> Scaleout:
    global _api_client_instance
    with _api_client_lock:
        if _api_client_instance is None:
            host, port = _default_host_port()
            _api_client_instance = Scaleout(
                host=host,
                port=port,
                access_token_provider=_api_token_function,
            )
        return _api_client_instance


class _ScaleoutProxy:
    """Proxy for backwards compatibility: `api_client.foo()` works."""

    def __getattr__(self, name: str):
        return getattr(_get_api_client(), name)

    def __call__(self, *args, **kwargs):
        return _get_api_client()(*args, **kwargs)


api_client: Scaleout = _ScaleoutProxy()

# --- Combiner context ---
_COMBINER_NAME: Optional[str] = None
_COMBINER_ID: Optional[str] = None


# combiner id can be useful e.g. for sharing attributes across sessions and combiners using the api client
def get_combiner_name() -> str:
    """Return the ID of the current combiner.

    The combiner ID is injected by the Scaleout runtime and is only
    available while code is executing in a combiner context. It can be
    used, for example, together with :data:`api_client` to share
    attributes or state across sessions and combiners.

    Returns:
        str: The identifier of the current combiner.

    Raises:
        RuntimeError: If the combiner ID has not been set yet, for
            example when called outside of a combiner context or before
            the runtime has initialised it.

    """
    if _COMBINER_NAME is None:
        raise RuntimeError("combiner_name not set.")
    return _COMBINER_NAME


def _resolve_combiner_id() -> Optional[str]:
    """Resolve combiner ID from name via the API."""
    try:
        result = api_client.get_combiners()
        for combiner in result.get("result", []):
            if combiner.get("name") == get_combiner_name():
                combiner_id = combiner.get("combiner_id")
                ScaleoutLogger().info(f"resolved combiner id: {combiner_id}")
                return combiner_id
        ScaleoutLogger().warning(f"Could not resolve combiner ID for name: {get_combiner_name()}")
    except Exception as e:
        ScaleoutLogger().warning(f"Failed to resolve combiner ID: {e}")
    return None


def print(*args, **kwargs):
    global _COMBINER_ID
    if _COMBINER_ID is None:
        _COMBINER_ID = _resolve_combiner_id()

    message = " ".join(str(a) for a in args)

    MAX_LEN = 255

    if len(message) > MAX_LEN:
        safe_message = message[:200] + f"... [truncated, total length={len(message)}]"
    else:
        safe_message = message

    ScaleoutLogger().info(message)

    api_client.add_status({"status": safe_message, "log_level": "INFO", "type": "PRINT", "sender": {"combiner_id": _COMBINER_ID}})
