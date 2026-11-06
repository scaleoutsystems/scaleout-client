"""Unit tests for CLI token handling logic in client_start_cmd.

Token priority in `client_start_cmd` (see client_cmd.py):
  1. enrollment token (exchanged for a refresh token)
  2. CLI `--token` value
  3. token cache
"""

import base64
import json
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, Mock
from click.testing import CliRunner

from scaleout.cli.client_cmd import client_start_cmd


def _make_jwt(payload: dict) -> str:
    """Build an unsigned JWT whose payload decodes to `payload` (signature not verified)."""
    def _b64(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{_b64({'alg': 'HS256', 'typ': 'JWT'})}.{_b64(payload)}.sig"


class TestCLITokenHandling(unittest.TestCase):
    """Test cases for token resolution in the CLI."""

    def setUp(self):
        self.runner = CliRunner()
        self.client_id = "test-client-id"
        self.cli_token = "cli-refresh-token"
        self.cached_refresh_token = "cached-refresh-token"
        self.cached_access_token = "cached-access-token"
        self.valid_expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    def _cached_data(self, refresh_token, access_token, expiry):
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expiry,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_cli_token_takes_precedence_over_cache(self, mock_cache_class, mock_client):
        """CLI --token should win over any cached token."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.load.return_value = self._cached_data(
            self.cached_refresh_token, self.cached_access_token, self.valid_expiry
        )
        mock_client.return_value.start = Mock()

        with self.runner.isolated_filesystem():
            self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.cli_token,
                '-u', 'http://localhost:8092',
            ])

        call_kwargs = mock_client.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], self.cli_token)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_falls_back_to_cache_when_no_cli_token(self, mock_cache_class, mock_client):
        """With no --token, the cached refresh token should be used."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.load.return_value = self._cached_data(
            self.cached_refresh_token, self.cached_access_token, self.valid_expiry
        )
        mock_client.return_value.start = Mock()

        with self.runner.isolated_filesystem():
            self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '-u', 'http://localhost:8092',
            ])

        call_kwargs = mock_client.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], self.cached_refresh_token)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_no_cli_token_no_cache_generates_uuid(self, mock_cache_class, mock_client):
        """Without a CLI token or cache, the command should still start (with a generated client_id)."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = False
        mock_cache.load.return_value = None
        mock_client.return_value.start = Mock()

        with self.runner.isolated_filesystem():
            self.runner.invoke(client_start_cmd, [
                '-u', 'http://localhost:8092',
            ])

        self.assertIsNotNone(mock_client.call_args)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_token_refresh_callback_saves_to_cache(self, mock_cache_class, mock_client):
        """The token-refresh callback passed to the client should persist tokens to the cache."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = False
        mock_client.return_value.start = Mock()

        with self.runner.isolated_filesystem():
            self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.cli_token,
                '-u', 'http://localhost:8092',
            ])

        callback = mock_client.call_args[1].get('token_refresh_callback')
        self.assertIsNotNone(callback)

        new_access = "new-access-token"
        new_refresh = "new-refresh-token"
        new_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        callback(new_access, new_refresh, new_expires)

        mock_cache.save.assert_called_once_with(new_access, new_refresh, new_expires)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_api_key_is_rejected_with_helpful_error(self, mock_cache_class, mock_client):
        """An API key must not be usable to start an edge client; it should fail with guidance."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = False
        mock_client.return_value.start = Mock()

        api_key = _make_jwt({"token_type": "api_key", "role": "client", "sub": "user-1"})

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', api_key,
                '-u', 'http://localhost:8092',
            ])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("scaleout client enroll", result.output.lower())
        mock_client.return_value.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
