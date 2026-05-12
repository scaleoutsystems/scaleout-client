"""Unit tests for CLI token handling logic in client_start_cmd."""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, Mock
from click.testing import CliRunner

from scaleout.cli.client_cmd import client_start_cmd


class TestCLITokenHandling(unittest.TestCase):
    """Test cases for smart token handling in CLI."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.client_id = "test-client-id"
        self.old_token = "old-refresh-token"
        self.new_token = "new-refresh-token"
        self.access_token = "test-access-token"
        
        # Future expiry (5 minutes from now)
        self.valid_expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        
        # Past expiry
        self.expired_expiry = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    def _create_cached_data(self, refresh_token, access_token, expiry):
        """Helper to create cached token data."""
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expiry,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_cli_token_same_as_cached_with_valid_access_token(self, mock_cache_class, mock_client):
        """Test CLI token matches cached token and access token is still valid."""
        # Setup mock cache
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.is_access_token_valid.return_value = True
        
        cached_data = self._create_cached_data(self.old_token, self.access_token, self.valid_expiry)
        mock_cache.load.return_value = cached_data
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.old_token,
                '-u', 'http://localhost:8092'
            ])
        
        # Verify access token was passed to client (not None)
        call_kwargs = mock_client.call_args[1]
        self.assertIsNotNone(call_kwargs.get('access_token'))
        self.assertEqual(call_kwargs['access_token'], self.access_token)
        
        # Verify output message
        self.assertIn("Using cached access token", result.output)
        self.assertIn("still valid", result.output)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_cli_token_same_as_cached_with_expired_access_token(self, mock_cache_class, mock_client):
        """Test CLI token matches cached token but access token is expired."""
        # Setup mock cache
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.is_access_token_valid.return_value = False
        
        cached_data = self._create_cached_data(self.old_token, self.access_token, self.expired_expiry)
        mock_cache.load.return_value = cached_data
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.old_token,
                '-u', 'http://localhost:8092'
            ])
        
        # Verify message about matching token with expired access token
        self.assertIn("CLI token matches cached token", result.output)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_cli_token_different_from_cached_with_valid_cached(self, mock_cache_class, mock_client):
        """Test CLI token differs from cached but cached token is still valid."""
        # Setup mock cache with different token
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.is_access_token_valid.return_value = True
        
        cached_data = self._create_cached_data(self.new_token, self.access_token, self.valid_expiry)
        mock_cache.load.return_value = cached_data
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.old_token,  # Different from cached
                '-u', 'http://localhost:8092'
            ])
        
        # Should prefer cached token over CLI token
        call_kwargs = mock_client.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], self.new_token)  # Uses cached, not CLI
        self.assertEqual(call_kwargs['access_token'], self.access_token)
        
        # Verify warning message
        self.assertIn("Warning", result.output)
        self.assertIn("CLI token differs from cached token", result.output)
        self.assertIn("Cached token is still valid", result.output)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_cli_token_different_from_cached_with_no_valid_cache(self, mock_cache_class, mock_client):
        """Test CLI token differs from cached and cached token is invalid."""
        # Setup mock cache with expired token
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        mock_cache.is_access_token_valid.return_value = False
        
        cached_data = self._create_cached_data(self.new_token, self.access_token, self.expired_expiry)
        mock_cache.load.return_value = cached_data
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.old_token,  # Different from cached
                '-u', 'http://localhost:8092'
            ])
        
        # Should use CLI token since cached is invalid
        call_kwargs = mock_client.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], self.old_token)  # Uses CLI token
        
        # Verify override message
        self.assertIn("overrides cached token", result.output)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_no_cli_token_uses_cached_token(self, mock_cache_class, mock_client):
        """Test that no CLI token uses cached token."""
        # Setup mock cache
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = True
        
        cached_data = self._create_cached_data(self.new_token, self.access_token, self.valid_expiry)
        mock_cache.load.return_value = cached_data
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '-u', 'http://localhost:8092'
            ])
        
        # Should use cached token
        call_kwargs = mock_client.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], self.new_token)
        
        # Verify loaded from cache message
        self.assertIn("Loaded refresh token from cache", result.output)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_no_cli_token_no_cache_generates_uuid(self, mock_cache_class, mock_client):
        """Test that no CLI token and no cache still works (generates UUID for client_id)."""
        # Setup mock cache with no cached data
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = False
        mock_cache.load.return_value = None
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '-u', 'http://localhost:8092'
            ])
        
        # Should not crash and client_id should be generated
        # (This test mainly ensures the code doesn't break without tokens)
        self.assertIsNotNone(mock_client.call_args)

    @patch('scaleout.cli.client_cmd.ImporterClient')
    @patch('scaleout.cli.client_cmd.TokenCache')
    def test_token_refresh_callback_saves_to_cache(self, mock_cache_class, mock_client):
        """Test that token refresh callback saves tokens to cache."""
        # Setup mock cache
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        mock_cache.exists.return_value = False
        
        # Mock client to prevent actual execution
        mock_client.return_value.start = Mock()
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(client_start_cmd, [
                '--client-id', self.client_id,
                '--token', self.old_token,
                '-u', 'http://localhost:8092'
            ])
        
        # Get the callback that was passed to the client
        call_kwargs = mock_client.call_args[1]
        callback = call_kwargs.get('token_refresh_callback')
        
        self.assertIsNotNone(callback)
        
        # Simulate callback being called
        test_access = "new-access-token"
        test_refresh = "new-refresh-token"
        test_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        
        callback(test_access, test_refresh, test_expires)
        
        # Verify cache.save was called
        mock_cache.save.assert_called_once_with(test_access, test_refresh, test_expires)


if __name__ == '__main__':
    unittest.main()
