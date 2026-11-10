"""Unit tests for the grpc_retry decorator's fail-fast paths.

These cover edge cases that are awkward to reproduce against a live combiner:
  - UNAUTHENTICATED with no refresh-capable credential must fail fast (no retry loop).
  - A manually closed channel (``_stop=True``) must re-raise without retry/reconnect.

They drive the decorator directly with a minimal fake handler, so no server or
network is involved.
"""

import unittest
from types import SimpleNamespace

import grpc

from scaleout.client.grpc_handler import RetryException, grpc_retry


class _FakeRpcError(grpc.RpcError):
    """A grpc.RpcError with the code()/details() accessors the decorator inspects."""

    def __init__(self, code, details="fake error"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeHandler:
    """Minimal stand-in for GrpcHandler exposing only what grpc_retry touches."""

    def __init__(self, login=None, stop=False):
        self._stop = stop
        self.on_reconnect = None
        self.client = SimpleNamespace(_login=login)
        self.reconnect_calls = 0

    def _reconnect_channel(self):
        self.reconnect_calls += 1


class TestGrpcRetryFailFast(unittest.TestCase):
    def test_unauthenticated_without_refresh_fails_fast(self):
        """UNAUTHENTICATED with no login (nothing to refresh) raises immediately, no retry."""
        calls = {"n": 0}

        @grpc_retry(max_retries=5, base_retry_interval=0.0)
        def always_unauth(self):
            calls["n"] += 1
            raise _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED)

        handler = _FakeHandler(login=None)
        with self.assertRaises(grpc.RpcError):
            always_unauth(handler)
        self.assertEqual(calls["n"], 1)  # called once, did not enter the retry loop
        self.assertEqual(handler.reconnect_calls, 0)

    def test_unauthenticated_with_api_key_fails_fast(self):
        """An API key cannot be refreshed, so UNAUTHENTICATED must not spin on retries."""
        calls = {"n": 0}

        @grpc_retry(max_retries=5, base_retry_interval=0.0)
        def always_unauth(self):
            calls["n"] += 1
            raise _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED)

        login = SimpleNamespace(ctype="api_key", _manager=None)
        handler = _FakeHandler(login=login)
        with self.assertRaises(grpc.RpcError):
            always_unauth(handler)
        self.assertEqual(calls["n"], 1)

    def test_unauthenticated_with_refresh_retries(self):
        """UNAUTHENTICATED with a refresh-capable login is transient and should retry."""
        calls = {"n": 0}

        @grpc_retry(max_retries=3, base_retry_interval=0.0)
        def always_unauth(self):
            calls["n"] += 1
            raise _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED)

        login = SimpleNamespace(ctype="client_refresh", _manager=object())
        handler = _FakeHandler(login=login)
        with self.assertRaises(RetryException):  # exhausts retries rather than bare-raising
            always_unauth(handler)
        self.assertEqual(calls["n"], 3)

    def test_manual_stop_does_not_retry_or_reconnect(self):
        """A manually closed channel (_stop=True) re-raises without retrying/reconnecting."""
        calls = {"n": 0}

        @grpc_retry(max_retries=5, base_retry_interval=0.0)
        def always_unavailable(self):
            calls["n"] += 1
            # UNAVAILABLE would normally trigger reconnect + retry.
            raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE)

        handler = _FakeHandler(stop=True)
        with self.assertRaises(grpc.RpcError):
            always_unavailable(handler)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(handler.reconnect_calls, 0)


if __name__ == "__main__":
    unittest.main()
