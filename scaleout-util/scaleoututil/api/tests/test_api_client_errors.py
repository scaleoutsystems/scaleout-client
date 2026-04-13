"""Unit tests for Scaleout error handling and edge cases."""

import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import requests

from scaleoututil.api.client import Scaleout


class TestScaleoutHTTPErrors(unittest.TestCase):
    """Test cases for HTTP error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.get')
    def test_get_client_404_not_found(self, mock_get):
        """Test handling of 404 Not Found response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Client not found"}
        mock_get.return_value = mock_response
        
        result = self.client.get_client("nonexistent-id")
        
        self.assertEqual(result, {"error": "Client not found"})

    @patch('requests.get')
    def test_get_clients_500_server_error(self, mock_get):
        """Test handling of 500 Internal Server Error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_get.return_value = mock_response
        
        result = self.client.get_clients()
        
        self.assertEqual(result, {"error": "Internal server error"})

    @patch('requests.get')
    def test_get_model_401_unauthorized(self, mock_get):
        """Test handling of 401 Unauthorized response."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_get.return_value = mock_response
        
        result = self.client.get_model("model-id")
        
        self.assertEqual(result, {"error": "Unauthorized"})

    @patch('requests.get')
    def test_get_session_403_forbidden(self, mock_get):
        """Test handling of 403 Forbidden response."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Forbidden"}
        mock_get.return_value = mock_response
        
        result = self.client.get_session(id="session-id")
        
        self.assertEqual(result, {"error": "Forbidden"})


class TestScaleoutNetworkErrors(unittest.TestCase):
    """Test cases for network-related errors."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.get')
    def test_connection_timeout(self, mock_get):
        """Test handling of connection timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        with self.assertRaises(requests.exceptions.Timeout):
            self.client.get_clients()

    @patch('requests.get')
    def test_connection_error(self, mock_get):
        """Test handling of connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.get_combiner("combiner-id")

    @patch('requests.get')
    def test_dns_resolution_failure(self, mock_get):
        """Test handling of DNS resolution failure."""
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "Failed to resolve hostname"
        )
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.get_controller_status()

    @patch('requests.put')
    def test_post_request_timeout(self, mock_put):
        """Test handling of timeout on PUT request (set_active_model uses PUT)."""
        mock_put.side_effect = requests.exceptions.Timeout("Request timed out")

        with self.assertRaises(requests.exceptions.Timeout):
            self.client.set_active_model("/tmp/model.npz")


class TestScaleoutInvalidResponses(unittest.TestCase):
    """Test cases for invalid or malformed API responses."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.get')
    def test_invalid_json_response(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError):
            self.client.get_models()

    @patch('requests.get')
    def test_empty_response_body(self, mock_get):
        """Test handling of empty response body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        result = self.client.get_packages()
        
        self.assertEqual(result, {})

    @patch('requests.get')
    def test_unexpected_response_format(self, mock_get):
        """Test handling of unexpected response format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}
        mock_get.return_value = mock_response
        
        result = self.client.get_active_model()
        
        # Should return the response even if format is unexpected
        self.assertEqual(result, {"unexpected": "format"})


class TestScaleoutFileOperations(unittest.TestCase):
    """Test cases for file download and upload operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_model_success(self, mock_file, mock_get):
        """Test successful model download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"model data"
        mock_get.return_value = mock_response
        
        result = self.client.download_model("model-id", "/tmp/model.npz")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Model downloaded successfully.")
        mock_file.assert_called_once_with("/tmp/model.npz", "wb")

    @patch('requests.get')
    def test_download_model_404(self, mock_get):
        """Test model download with 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.client.download_model("nonexistent-model", "/tmp/model.npz")
        
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Failed to download model.")

    @patch('requests.get')
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_download_model_permission_error(self, mock_file, mock_get):
        """Test model download with file write permission error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"model data"
        mock_get.return_value = mock_response
        
        with self.assertRaises(PermissionError):
            self.client.download_model("model-id", "/root/model.npz")

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_package_success(self, mock_file, mock_get):
        """Test successful package download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"package data"
        mock_get.return_value = mock_response
        
        result = self.client.download_package("/tmp/package.tar.gz")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Package downloaded successfully.")

    @patch('requests.get')
    def test_download_package_failure(self, mock_get):
        """Test package download failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = self.client.download_package("/tmp/package.tar.gz")
        
        self.assertFalse(result["success"])

    @patch('requests.post')
    @patch('requests.put')
    def test_set_active_model_npz(self, mock_put, mock_post):
        """Test setting active model with .npz file."""
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"success": True}
        mock_post.return_value = mock_post_response

        mock_put_response = MagicMock()
        mock_put.return_value = mock_put_response

        with patch.object(self.client, '_perform_chunked_upload', return_value='fake-token') as mock_upload:
            result = self.client.set_active_model("/tmp/model.npz")
            mock_upload.assert_called_once_with("/tmp/model.npz")

        self.assertEqual(result, {"success": True})

    @patch('requests.post')
    @patch('requests.put')
    def test_set_active_model_bin(self, mock_put, mock_post):
        """Test setting active model with .bin file."""
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"success": True}
        mock_post.return_value = mock_post_response

        mock_put_response = MagicMock()
        mock_put.return_value = mock_put_response

        with patch.object(self.client, '_perform_chunked_upload', return_value='fake-token'):
            result = self.client.set_active_model("/tmp/model.bin")

        self.assertEqual(result, {"success": True})

    @patch('requests.put')
    def test_set_active_model_file_not_found(self, mock_put):
        """Test setting active model with non-existent file."""
        mock_put.return_value = MagicMock()

        with self.assertRaises(FileNotFoundError):
            self.client.set_active_model("/nonexistent/model.npz")


class TestScaleoutSessionMethods(unittest.TestCase):
    """Test cases for session-related methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.post')
    @patch('requests.get')
    def test_start_session_no_models(self, mock_get, mock_post):
        """Test starting session when no models exist."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"result": []}
        mock_get.return_value = mock_get_response
        
        result = self.client.start_session(name="test-session")
        
        self.assertIn("message", result)
        self.assertIn("No models found", result["message"])

    @patch('requests.post')
    @patch('requests.get')
    def test_start_session_with_model_id(self, mock_get, mock_post):
        """Test starting session with explicit model_id."""
        mock_post_response1 = MagicMock()
        mock_post_response1.status_code = 201
        mock_post_response1.json.return_value = {"session_id": "session-123"}
        
        mock_post_response2 = MagicMock()
        mock_post_response2.status_code = 200
        mock_post_response2.json.return_value = {"success": True, "session_id": "session-123"}
        
        mock_post.side_effect = [mock_post_response1, mock_post_response2]
        
        result = self.client.start_session(name="test-session", model_id="model-123")
        
        self.assertIn("session_id", result)
        self.assertEqual(result["session_id"], "session-123")

    @patch('requests.get')
    def test_get_session_by_name_not_found(self, mock_get):
        """Test getting session by name when not found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": []}
        mock_get.return_value = mock_response
        
        result = self.client.get_session(name="nonexistent-session")
        
        self.assertIn("message", result)
        self.assertIn("Session not found", result["message"])

    @patch('requests.get')
    def test_get_session_no_id_no_name(self, mock_get):
        """Test getting session without id or name."""
        result = self.client.get_session()
        
        self.assertIn("message", result)
        self.assertIn("No id or name provided", result["message"])

    @patch('requests.get')
    def test_session_is_finished_true(self, mock_get):
        """Test checking if session is finished (true case)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "finished", "session_id": "session-123"}
        mock_get.return_value = mock_response
        
        result = self.client.session_is_finished("session-123")
        
        self.assertTrue(result)

    @patch('requests.get')
    def test_session_is_finished_false(self, mock_get):
        """Test checking if session is finished (false case)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "running", "session_id": "session-123"}
        mock_get.return_value = mock_response
        
        result = self.client.session_is_finished("session-123")
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_get_session_status_error(self, mock_get):
        """Test getting session status when session not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_response
        
        result = self.client.get_session_status("nonexistent-id")
        
        self.assertEqual(result, "Could not retrieve session status.")


class TestScaleoutFilteringAndPagination(unittest.TestCase):
    """Test cases for filtering and pagination parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch('requests.get')
    def test_get_clients_with_limit(self, mock_get):
        """Test getting clients with n_max limit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": ["client1", "client2"]}
        mock_get.return_value = mock_response
        
        result = self.client.get_clients(n_max=10)
        
        # Verify X-Limit header was set
        call_headers = mock_get.call_args[1]['headers']
        self.assertEqual(call_headers.get('X-Limit'), '10')

    @patch('requests.get')
    def test_get_active_clients_with_combiner_filter(self, mock_get):
        """Test getting active clients filtered by combiner."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": ["client1"]}
        mock_get.return_value = mock_response
        
        result = self.client.get_active_clients(combiner_id="combiner-123")
        
        # Verify params were set
        call_params = mock_get.call_args[1]['params']
        self.assertEqual(call_params['combiner'], "combiner-123")
        self.assertEqual(call_params['status'], "online")

    @patch('requests.get')
    def test_get_models_with_session_filter(self, mock_get):
        """Test getting models filtered by session_id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": ["model1", "model2"]}
        mock_get.return_value = mock_response
        
        result = self.client.get_models(session_id="session-123")
        
        # Verify params were set
        call_params = mock_get.call_args[1]['params']
        self.assertEqual(call_params['session_id'], "session-123")

    @patch('requests.get')
    def test_get_rounds_with_filter(self, mock_get):
        """Test getting rounds with filter parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": ["round1"]}
        mock_get.return_value = mock_response
        
        filter_dict = {"session_id": "session-123", "round_id": "5"}
        result = self.client.get_rounds(filter=filter_dict)
        
        # Verify filter params were converted to strings and set
        call_params = mock_get.call_args[1]['params']
        self.assertEqual(call_params['session_id'], "session-123")
        self.assertEqual(call_params['round_id'], "5")


class TestScaleoutValidation(unittest.TestCase):
    """Test cases for input validation and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_get_active_model_empty_result(self):
        """Test get_active_model when no models exist."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": []}
            mock_get.return_value = mock_response
            
            result = self.client.get_active_model()
            
            # Should return the response even with empty result
            self.assertEqual(result, {"result": []})

    def test_get_model_trail_no_id(self):
        """Test get_model_trail without id falls back to active model."""
        with patch('requests.get') as mock_get:
            # Mock get_active_model response
            mock_response1 = MagicMock()
            mock_response1.status_code = 200
            mock_response1.json.return_value = {
                "result": [{"model_id": "active-model-123"}]
            }
            
            # Mock get_model_trail response
            mock_response2 = MagicMock()
            mock_response2.status_code = 200
            mock_response2.json.return_value = {"result": ["model1", "model2"]}
            
            mock_get.side_effect = [mock_response1, mock_response1, mock_response2]
            
            result = self.client.get_model_trail()
            
            self.assertIn("result", result)

    @patch('requests.post')
    def test_run_custom_command_adds_custom_prefix(self, mock_post):
        """Test that run_custom_command adds Custom_ prefix if missing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        result = self.client.run_custom_command("MyCommand")
        
        # Verify Custom_ prefix was added
        call_data = mock_post.call_args[1]['json']
        self.assertEqual(call_data['command_type'], "Custom_MyCommand")

    @patch('requests.post')
    def test_run_custom_command_preserves_custom_prefix(self, mock_post):
        """Test that run_custom_command preserves existing Custom_ prefix."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        result = self.client.run_custom_command("Custom_MyCommand")
        
        # Verify Custom_ prefix was not duplicated
        call_data = mock_post.call_args[1]['json']
        self.assertEqual(call_data['command_type'], "Custom_MyCommand")


if __name__ == "__main__":
    unittest.main()
