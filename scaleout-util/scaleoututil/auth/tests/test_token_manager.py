"""Unit tests for TokenManager."""

import requests
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from scaleoututil.auth.token_manager import TokenManager


class TestTokenManager(unittest.TestCase):
    """Test cases for TokenManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.access_token = "test-access-token"
        self.refresh_token = "test-refresh-token"
        self.token_endpoint = "http://localhost:8092/api/v1/auth/refresh"

    def test_init_with_refresh_token(self):
        """Test TokenManager initialization with refresh token."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=3600
        )
        
        self.assertEqual(manager._access_token, self.access_token)
        self.assertEqual(manager._refresh_token, self.refresh_token)
        self.assertEqual(manager._token_endpoint, self.token_endpoint)
        self.assertIsNotNone(manager._token_expires_at)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_init_with_only_refresh_token(self, mock_post):
        """Test TokenManager initialization with only refresh token performs initial exchange."""
        new_access_token = "new-access-token"
        
        # Mock the token refresh response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "refresh_token": self.refresh_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        manager = TokenManager(
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint
        )
        
        # Should have performed refresh to get access token
        self.assertEqual(manager._access_token, new_access_token)
        self.assertEqual(manager._refresh_token, self.refresh_token)
        mock_post.assert_called_once()

    def test_get_access_token_no_refresh_needed(self):
        """Test getting access token when no refresh is needed."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=7200  # 2 hours, well beyond buffer
        )
        
        token = manager.get_access_token()
        self.assertEqual(token, self.access_token)

    def test_should_refresh_token_within_buffer(self):
        """Test that token should refresh when within buffer time."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20  # 20 seconds, less than 30 second buffer
        )
        
        self.assertTrue(manager._should_refresh_token())

    def test_should_not_refresh_token_outside_buffer(self):
        """Test that token should not refresh when outside buffer time."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=7200  # 2 hours
        )
        
        self.assertFalse(manager._should_refresh_token())

    def test_should_not_refresh_without_refresh_token(self):
        """Test that refresh is skipped when no refresh token is available."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=None,
            token_endpoint=self.token_endpoint,
            expires_in=200
        )
        
        self.assertFalse(manager._should_refresh_token())

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_perform_token_refresh_success(self, mock_post):
        """Test successful token refresh."""
        new_access_token = "new-access-token"
        new_refresh_token = "new-refresh-token"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=100
        )
        
        manager._perform_token_refresh()
        
        self.assertEqual(manager._access_token, new_access_token)
        self.assertEqual(manager._refresh_token, new_refresh_token)
        mock_post.assert_called_once()

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_perform_token_refresh_unauthorized(self, mock_post):
        """Test token refresh failure with 401 Unauthorized."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=100
        )
        
        with self.assertRaises(RuntimeError) as context:
            manager._perform_token_refresh()
        
        self.assertIn("invalid or expired", str(context.exception))

    def test_perform_token_refresh_no_refresh_token(self):
        """Test that refresh fails gracefully when no refresh token is available."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=None,
            token_endpoint=self.token_endpoint,
            expires_in=100
        )
        
        # Should return early without raising
        manager._perform_token_refresh()
        # Token should remain unchanged
        self.assertEqual(manager._access_token, self.access_token)

    def test_perform_token_refresh_no_endpoint(self):
        """Test that refresh fails gracefully when no token endpoint is configured."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=None,
            expires_in=100
        )
        
        # Should return early without raising
        manager._perform_token_refresh()
        # Token should remain unchanged
        self.assertEqual(manager._access_token, self.access_token)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_get_access_token_with_auto_refresh(self, mock_post):
        """Test that get_access_token automatically refreshes when needed."""
        new_access_token = "new-access-token"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20  # 20 seconds, within 30 second buffer, will trigger refresh
        )
        
        token = manager.get_access_token()
        
        self.assertEqual(token, new_access_token)
        mock_post.assert_called_once()

    def test_update_tokens_manually(self):
        """Test manual token update."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint
        )
        
        new_access = "new-access"
        new_refresh = "new-refresh"
        
        manager.update_tokens(
            access_token=new_access,
            refresh_token=new_refresh,
            expires_in=1800
        )
        
        self.assertEqual(manager._access_token, new_access)
        self.assertEqual(manager._refresh_token, new_refresh)

    def test_get_auth_header(self):
        """Test getting authorization header."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=7200
        )
        
        header = manager.get_auth_header()
        
        self.assertIn("Authorization", header)
        self.assertEqual(header["Authorization"], f"Bearer {self.access_token}")

    def test_is_token_expired(self):
        """Test token expiration check."""
        # Not expired
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=3600
        )
        self.assertFalse(manager.is_token_expired())
        
        # Expired
        manager._token_expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        self.assertTrue(manager.is_token_expired())

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_concurrent_token_refresh(self, mock_post):
        """Test that concurrent refresh attempts are handled safely."""
        new_access_token = "new-access-token"
        
        # Simulate slow refresh (to ensure multiple threads hit the refresh logic)
        def slow_refresh(*args, **kwargs):
            time.sleep(0.1)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": new_access_token,
                "expires_in": 3600
            }
            return mock_response
        
        mock_post.side_effect = slow_refresh
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20  # Within 30 second buffer
        )
        
        # Start multiple threads trying to get token
        results = []
        
        def get_token():
            token = manager.get_access_token()
            results.append(token)
        
        threads = [threading.Thread(target=get_token) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All threads should get the same new token
        self.assertEqual(len(results), 5)
        self.assertTrue(all(token == new_access_token for token in results))
        
        # Refresh should only happen once despite multiple threads
        self.assertEqual(mock_post.call_count, 1)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_refresh_with_network_error(self, mock_post):
        """Test handling of network errors during refresh."""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=-1  # already expired, so there is no valid token to fall back to
        )
        
        with self.assertRaises(RuntimeError) as context:
            manager.get_access_token()
        
        self.assertIn("Token refresh failed", str(context.exception))

    def test_token_manager_without_refresh_capability(self):
        """Test TokenManager behavior without refresh token."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=None,
            token_endpoint=self.token_endpoint,
            expires_in=100  # Would normally trigger refresh
        )
        
        # Should return current token without attempting refresh
        token = manager.get_access_token()
        self.assertEqual(token, self.access_token)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_callback_invoked_on_token_refresh(self, mock_post):
        """Test that on_token_refresh callback is called with correct parameters."""
        new_access_token = "new-access-token"
        new_refresh_token = "new-refresh-token"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        # Create mock callback
        callback_args = []
        def mock_callback(access_token, refresh_token, expires_at):
            callback_args.append({
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at
            })
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20,  # Within buffer, will trigger refresh
            on_token_refresh=mock_callback
        )
        
        # Trigger refresh
        manager.get_access_token()
        
        # Verify callback was called exactly once
        self.assertEqual(len(callback_args), 1)
        
        # Verify callback received correct tokens
        self.assertEqual(callback_args[0]["access_token"], new_access_token)
        self.assertEqual(callback_args[0]["refresh_token"], new_refresh_token)
        
        # Verify expires_at is a datetime object
        self.assertIsInstance(callback_args[0]["expires_at"], datetime)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_callback_invoked_on_initial_token_exchange(self, mock_post):
        """Test that callback is called during initial token exchange (no access token provided)."""
        new_access_token = "initial-access-token"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "refresh_token": self.refresh_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        # Create mock callback
        callback_called = []
        def mock_callback(access_token, refresh_token, expires_at):
            callback_called.append(True)
        
        # Initialize with only refresh token (triggers initial exchange)
        manager = TokenManager(
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            on_token_refresh=mock_callback
        )
        
        # Verify callback was called during initialization
        self.assertEqual(len(callback_called), 1)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_callback_exception_does_not_fail_refresh(self, mock_post):
        """Test that exceptions in callback don't prevent token refresh from completing."""
        new_access_token = "new-access-token"
        new_refresh_token = "new-refresh-token"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "expires_in": 3600
        }
        mock_post.return_value = mock_response
        
        # Create callback that raises exception
        def failing_callback(access_token, refresh_token, expires_at):
            raise ValueError("Callback error")
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20,  # Within buffer, will trigger refresh
            on_token_refresh=failing_callback
        )
        
        # Refresh should succeed despite callback exception
        token = manager.get_access_token()
        
        # Verify token was updated successfully
        self.assertEqual(token, new_access_token)
        self.assertEqual(manager._access_token, new_access_token)
        self.assertEqual(manager._refresh_token, new_refresh_token)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_callback_not_invoked_when_refresh_not_needed(self, mock_post):
        """Test that callback is not called when refresh is not needed."""
        callback_called = []
        def mock_callback(access_token, refresh_token, expires_at):
            callback_called.append(True)
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=7200,  # Well beyond buffer, no refresh needed
            on_token_refresh=mock_callback
        )
        
        # Get token (should not trigger refresh)
        manager.get_access_token()
        
        # Verify callback was not called
        self.assertEqual(len(callback_called), 0)
        
        # Verify no HTTP request was made
        mock_post.assert_not_called()

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_callback_receives_updated_expiration(self, mock_post):
        """Test that callback receives the updated expiration time from refreshed token."""
        new_access_token = "new-access-token"
        expires_in = 1800  # 30 minutes
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_access_token,
            "expires_in": expires_in
        }
        mock_post.return_value = mock_response
        
        callback_expires_at = []
        def mock_callback(access_token, refresh_token, expires_at):
            callback_expires_at.append(expires_at)
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=20,
            on_token_refresh=mock_callback
        )
        
        # Trigger refresh
        manager.get_access_token()
        
        # Verify callback received an expiration time
        self.assertEqual(len(callback_expires_at), 1)
        expires_at = callback_expires_at[0]
        
        # Verify expiration is in the future (approximately expires_in seconds)
        now = datetime.now(timezone.utc)
        time_until_expiry = (expires_at - now).total_seconds()
        
        # Should be roughly 1800 seconds (with some tolerance for test execution time)
        self.assertGreater(time_until_expiry, 1700)
        self.assertLess(time_until_expiry, 1900)

    def test_callback_none_does_not_error(self):
        """Test that TokenManager works correctly when callback is None."""
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=3600,
            on_token_refresh=None  # Explicitly set to None
        )
        
        # Should initialize without error
        self.assertEqual(manager._access_token, self.access_token)
        self.assertIsNone(manager._on_token_refresh)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_init_refresh_token_failure_provides_helpful_error(self, mock_post):
        """Test that initial token exchange failure provides helpful error message."""
        # Mock failed refresh response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"error":"Token refresh failed"}'
        mock_post.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            TokenManager(
                refresh_token=self.refresh_token,
                token_endpoint=self.token_endpoint
            )
        
        # Verify error message is helpful
        error_msg = str(context.exception)
        self.assertIn("Refresh token is invalid or expired", error_msg)
        self.assertIn("log in again", error_msg)

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_init_refresh_token_failure_logs_hint(self, mock_post):
        """Test that initial token exchange failure logs hint about cached token."""
        # Mock failed refresh response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"error":"Token refresh failed"}'
        mock_post.return_value = mock_response
        
        with patch('scaleoututil.auth.token_manager.ScaleoutLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            try:
                TokenManager(
                    refresh_token=self.refresh_token,
                    token_endpoint=self.token_endpoint
                )
            except RuntimeError:
                pass
            
            # Verify helpful log messages were created
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            hint_logged = any("cached token" in call.lower() and "--token" in call.lower() 
                            for call in info_calls)
            self.assertTrue(hint_logged, "Expected hint about cached token to be logged")

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_get_access_token_with_valid_cached_token(self, mock_post):
        """Test that get_access_token doesn't refresh when token is still valid."""
        # Create manager with access token that won't expire soon
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=3600  # 1 hour - well beyond refresh buffer
        )
        
        # Get access token should return existing token without refresh
        token = manager.get_access_token()
        
        self.assertEqual(token, self.access_token)
        mock_post.assert_not_called()  # Should not trigger refresh

    @patch('scaleoututil.auth.token_manager.requests.post')
    def test_refresh_token_invalid_provides_helpful_error(self, mock_post):
        """Test that token refresh failure with 401 provides helpful error message."""
        # Mock failed refresh response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"error":"Token refresh failed","details":"Token endpoint returned an error","status":401}'
        mock_post.return_value = mock_response
        
        manager = TokenManager(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            token_endpoint=self.token_endpoint,
            expires_in=1  # Will expire immediately
        )
        
        # Wait a moment to ensure expiry
        time.sleep(2)
        
        with self.assertRaises(RuntimeError) as context:
            manager.get_access_token()
        
        # Verify error message is helpful
        error_msg = str(context.exception)
        self.assertIn("Refresh token is invalid or expired", error_msg)
        self.assertIn("log in again", error_msg)


if __name__ == '__main__':
    unittest.main()
