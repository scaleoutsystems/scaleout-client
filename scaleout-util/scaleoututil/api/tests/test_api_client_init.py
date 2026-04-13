"""Unit tests for Scaleout initialization and URL handling."""

import os
import unittest
from unittest.mock import patch, MagicMock

import requests

from scaleoututil.api.client import Scaleout


class TestScaleoutInitialization(unittest.TestCase):
    """Test cases for Scaleout initialization scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        # Disable token manager for initialization tests
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        
        # Mock TokenCache to prevent file operations
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_protocol_in_host_https(self):
        """Test initialization with https:// protocol in host URL."""
        client = Scaleout(host="https://example.com")
        
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 443)

    def test_protocol_in_host_http(self):
        """Test initialization with http:// protocol in host URL."""
        client = Scaleout(host="http://example.com")
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 80)

    def test_protocol_in_host_with_custom_port(self):
        """Test initialization with protocol and custom port in host URL."""
        client = Scaleout(host="https://example.com:8080")
        
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 8080)

    def test_explicit_secure_true(self):
        """Test initialization with explicit secure=True parameter."""
        client = Scaleout(host="example.com", secure=True)
        
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 443)

    def test_explicit_secure_false(self):
        """Test initialization with explicit secure=False parameter."""
        client = Scaleout(host="example.com", secure=False)
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 80)

    def test_port_443_auto_detects_https(self):
        """Test that port 443 automatically enables HTTPS."""
        client = Scaleout(host="example.com", port=443)
        
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 443)

    def test_explicit_port_with_secure_true(self):
        """Test initialization with custom port and secure=True."""
        client = Scaleout(host="example.com", port=8443, secure=True)
        
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 8443)

    def test_explicit_port_with_secure_false(self):
        """Test initialization with custom port and secure=False."""
        client = Scaleout(host="example.com", port=8080, secure=False)
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 8080)

    def test_no_port_no_secure_defaults_to_http_80(self):
        """Test that no port and no secure parameter defaults to HTTP on port 80."""
        client = Scaleout(host="example.com")
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 80)

    def test_protocol_overrides_secure_parameter(self):
        """Test that protocol in host URL overrides secure parameter."""
        # Reset the singleton to get a fresh logger
        from scaleoututil.logging import ScaleoutLogger as LoggerClass
        LoggerClass._instance = None
        
        # Create the logger instance and patch its warning method
        logger_instance = LoggerClass()
        with patch.object(logger_instance, 'warning') as mock_warning:
            client = Scaleout(host="https://example.com", secure=False)
            
            # Verify warning message was logged
            mock_warning.assert_called_once()
            self.assertIn("Both protocol in host and secure parameter", mock_warning.call_args[0][0])
            
            # Protocol should override secure parameter
            self.assertEqual(client.host, "example.com")
            self.assertTrue(client.secure)
            self.assertEqual(client.port, 443)

    def test_http_protocol_overrides_secure_true(self):
        """Test that http:// protocol overrides secure=True parameter."""
        # Reset the singleton to get a fresh logger
        from scaleoututil.logging import ScaleoutLogger as LoggerClass
        LoggerClass._instance = None
        
        # Create the logger instance and patch its warning method
        logger_instance = LoggerClass()
        with patch.object(logger_instance, 'warning') as mock_warning:
            client = Scaleout(host="http://example.com", secure=True)
            
            # Verify warning message was logged
            mock_warning.assert_called_once()
            
            # Protocol should override secure parameter
            self.assertEqual(client.host, "example.com")
            self.assertFalse(client.secure)
            self.assertEqual(client.port, 80)

    def test_custom_port_no_secure_defaults_to_http(self):
        """Test that custom port without secure parameter defaults to HTTP."""
        client = Scaleout(host="example.com", port=9000)
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 9000)

    def test_port_443_with_explicit_secure_false(self):
        """Test port 443 with explicit secure=False (secure parameter wins)."""
        client = Scaleout(host="example.com", port=443, secure=False)
        
        self.assertEqual(client.host, "example.com")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 443)


class TestScaleoutURLConstruction(unittest.TestCase):
    """Test cases for Scaleout URL construction."""

    def setUp(self):
        """Set up test fixtures."""
        # Disable token manager for URL tests
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        
        # Mock TokenCache to prevent file operations
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_get_url_https_with_port(self):
        """Test URL construction for HTTPS with explicit port."""
        client = Scaleout(host="example.com", port=8443, secure=True)
        url = client._get_url("api/test")
        
        self.assertEqual(url, "https://example.com:8443/api/test")

    def test_get_url_http_with_port(self):
        """Test URL construction for HTTP with explicit port."""
        client = Scaleout(host="example.com", port=8080, secure=False)
        url = client._get_url("api/test")
        
        self.assertEqual(url, "http://example.com:8080/api/test")

    def test_get_url_https_default_port(self):
        """Test URL construction for HTTPS with default port 443."""
        client = Scaleout(host="example.com", secure=True)
        url = client._get_url("api/test")
        
        self.assertEqual(url, "https://example.com:443/api/test")

    def test_get_url_http_default_port(self):
        """Test URL construction for HTTP with default port 80."""
        client = Scaleout(host="example.com", secure=False)
        url = client._get_url("api/test")
        
        self.assertEqual(url, "http://example.com:80/api/test")

    def test_get_url_api_v1(self):
        """Test URL construction for API v1 endpoints."""
        client = Scaleout(host="example.com", secure=True)
        url = client._get_url_api_v1("clients/test")
        
        self.assertEqual(url, "https://example.com:443/api/v1/clients/test")


class TestScaleoutEdgeCases(unittest.TestCase):
    """Test cases for edge cases and special scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        # Disable token manager
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        
        # Mock TokenCache
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_localhost_defaults(self):
        """Test initialization with localhost."""
        client = Scaleout(host="localhost")
        
        self.assertEqual(client.host, "localhost")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 80)

    def test_localhost_with_port(self):
        """Test initialization with localhost and port."""
        client = Scaleout(host="localhost", port=8092, secure=False)
        
        self.assertEqual(client.host, "localhost")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 8092)

    def test_ip_address_with_port(self):
        """Test initialization with IP address and port."""
        client = Scaleout(host="192.168.1.100", port=9000, secure=False)
        
        self.assertEqual(client.host, "192.168.1.100")
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 9000)

    def test_complex_domain_name(self):
        """Test initialization with complex domain name."""
        client = Scaleout(host="wandering-math.scylla-atlas.ts.net", port=443)
        
        self.assertEqual(client.host, "wandering-math.scylla-atlas.ts.net")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 443)

    def test_verify_parameter(self):
        """Test that verify parameter is properly set."""
        client = Scaleout(host="example.com", verify=True)
        self.assertTrue(client.verify)
        
        client = Scaleout(host="example.com", verify=False)
        self.assertFalse(client.verify)

    def test_port_in_url_and_port_parameter_conflict(self):
        """Test that port parameter overrides port in URL with warning."""
        with patch('scaleoututil.api.client.ScaleoutLogger') as mock_logger:
            client = Scaleout(host="example.com:8080", port=9000)
            
            # Verify warning was logged
            mock_logger.return_value.warning.assert_called_once()
            warning_msg = mock_logger.return_value.warning.call_args[0][0]
            self.assertIn("Both port in host URL", warning_msg)
            self.assertIn("8080", warning_msg)
            self.assertIn("9000", warning_msg)
            
            # Port parameter should win
            self.assertEqual(client.host, "example.com")
            self.assertEqual(client.port, 9000)

    def test_port_in_url_and_same_port_parameter_no_warning(self):
        """Test that same port in URL and parameter doesn't trigger warning."""
        with patch('scaleoututil.api.client.ScaleoutLogger') as mock_logger:
            client = Scaleout(host="example.com:8080", port=8080)
            
            # No warning should be logged for matching ports
            mock_logger.return_value.warning.assert_not_called()
            
            self.assertEqual(client.host, "example.com")
            self.assertEqual(client.port, 8080)

    def test_port_in_url_no_port_parameter(self):
        """Test that port from URL is used when no port parameter provided."""
        client = Scaleout(host="example.com:8080")
        
        self.assertEqual(client.host, "example.com")
        self.assertEqual(client.port, 8080)


class TestScaleoutAuthScheme(unittest.TestCase):
    """Test cases for auth scheme handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()
        # Clean up env vars
        if 'SCALEOUT_AUTH_SCHEME' in os.environ:
            del os.environ['SCALEOUT_AUTH_SCHEME']

    def test_default_auth_scheme(self):
        """Test that auth scheme defaults to 'Bearer'."""
        client = Scaleout(host="example.com")
        self.assertEqual(client.auth_scheme, "Bearer")

    def test_explicit_auth_scheme(self):
        """Test explicit auth_scheme parameter."""
        client = Scaleout(host="example.com", auth_scheme="Token")
        self.assertEqual(client.auth_scheme, "Token")

    def test_auth_scheme_from_env_var(self):
        """Test auth scheme from SCALEOUT_AUTH_SCHEME env var."""
        os.environ['SCALEOUT_AUTH_SCHEME'] = "JWT"
        client = Scaleout(host="example.com")
        self.assertEqual(client.auth_scheme, "JWT")

    def test_auth_scheme_parameter_overrides_env_var(self):
        """Test that explicit parameter overrides env var."""
        os.environ['SCALEOUT_AUTH_SCHEME'] = "JWT"
        client = Scaleout(host="example.com", auth_scheme="Bearer")
        self.assertEqual(client.auth_scheme, "Bearer")


class TestScaleoutTokenEndpoint(unittest.TestCase):
    """Test cases for token endpoint construction."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_token_endpoint_constructed_https(self):
        """Test token endpoint is constructed correctly for HTTPS."""
        client = Scaleout(host="example.com", secure=True, token="test-token")
        
        # Verify TokenManager was called with correct endpoint
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['token_endpoint'], "https://example.com:443/api/auth/refresh")

    def test_token_endpoint_constructed_http(self):
        """Test token endpoint is constructed correctly for HTTP."""
        client = Scaleout(host="example.com", secure=False, token="test-token")
        
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['token_endpoint'], "http://example.com:80/api/auth/refresh")

    def test_token_endpoint_with_custom_port(self):
        """Test token endpoint with custom port."""
        client = Scaleout(host="example.com", port=8443, secure=True, token="test-token")
        
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['token_endpoint'], "https://example.com:8443/api/auth/refresh")

    def test_explicit_token_endpoint_parameter(self):
        """Test that explicit token_endpoint parameter is used."""
        custom_endpoint = "https://auth.example.com/refresh"
        client = Scaleout(host="example.com", token="test-token", token_endpoint=custom_endpoint)
        
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['token_endpoint'], custom_endpoint)


class TestScaleoutConfusingCombinations(unittest.TestCase):
    """Test cases for confusing or conflicting protocol/port combinations."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_https_url_with_port_80(self):
        """Test https:// URL with typically HTTP port 80."""
        client = Scaleout(host="https://example.com:80")
        
        # Protocol should win, but port should be from URL
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 80)

    def test_http_url_with_port_443(self):
        """Test http:// URL with typically HTTPS port 443."""
        client = Scaleout(host="http://example.com:443")
        
        # Protocol should win, but port should be from URL
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 443)

    def test_port_443_with_explicit_secure_false(self):
        """Test port 443 with explicit secure=False (user override)."""
        client = Scaleout(host="example.com", port=443, secure=False)
        
        # Explicit secure parameter should be respected
        self.assertFalse(client.secure)
        self.assertEqual(client.port, 443)

    def test_port_80_with_explicit_secure_true(self):
        """Test port 80 with explicit secure=True."""
        client = Scaleout(host="example.com", port=80, secure=True)
        
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 80)


class TestScaleoutEdgeCasesAndMalformed(unittest.TestCase):
    """Test cases for edge cases and potentially malformed inputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()

    def test_host_with_trailing_slash(self):
        """Test host with trailing slash."""
        client = Scaleout(host="example.com/")
        
        # Should handle gracefully
        self.assertIn("example.com", client.host)

    def test_url_with_path(self):
        """Test URL with path component."""
        client = Scaleout(host="https://example.com/api/v1")
        
        # Path should be stripped by parse_url
        self.assertEqual(client.host, "example.com")
        self.assertTrue(client.secure)

    def test_ipv4_address(self):
        """Test with IPv4 address."""
        client = Scaleout(host="192.168.1.100", port=8080)
        
        self.assertEqual(client.host, "192.168.1.100")
        self.assertEqual(client.port, 8080)

    def test_ipv4_address_in_url(self):
        """Test with IPv4 address in URL."""
        client = Scaleout(host="https://192.168.1.100:8443")
        
        self.assertEqual(client.host, "192.168.1.100")
        self.assertTrue(client.secure)
        self.assertEqual(client.port, 8443)

    def test_localhost_variations(self):
        """Test various localhost formats."""
        client1 = Scaleout(host="localhost")
        self.assertEqual(client1.host, "localhost")
        
        client2 = Scaleout(host="127.0.0.1")
        self.assertEqual(client2.host, "127.0.0.1")


class TestScaleoutEnvironmentVariables(unittest.TestCase):
    """Test cases for environment variable handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.token_patch = patch('scaleoututil.api.client.TokenManager')
        self.mock_token_manager = self.token_patch.start()
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False
        
        # Save original env vars
        self.original_auth_token = os.environ.get('SCALEOUT_AUTH_TOKEN')
        self.original_auth_scheme = os.environ.get('SCALEOUT_AUTH_SCHEME')
        self.original_cache_dir = os.environ.get('SCALEOUT_TOKEN_CACHE_DIR')

    def tearDown(self):
        """Clean up test fixtures."""
        self.token_patch.stop()
        self.cache_patch.stop()
        
        # Restore original env vars
        for key, value in [
            ('SCALEOUT_AUTH_TOKEN', self.original_auth_token),
            ('SCALEOUT_AUTH_SCHEME', self.original_auth_scheme),
            ('SCALEOUT_TOKEN_CACHE_DIR', self.original_cache_dir)
        ]:
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value

    def test_token_from_env_var(self):
        """Test token is loaded from SCALEOUT_AUTH_TOKEN env var."""
        os.environ['SCALEOUT_AUTH_TOKEN'] = "env-token"
        client = Scaleout(host="example.com")
        
        # TokenManager should be called with token from env
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], "env-token")

    def test_token_parameter_overrides_env_var(self):
        """Test that explicit token parameter overrides env var."""
        os.environ['SCALEOUT_AUTH_TOKEN'] = "env-token"
        client = Scaleout(host="example.com", token="param-token")
        
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        self.assertEqual(call_kwargs['refresh_token'], "param-token")

    def test_token_with_scheme_prefix_is_cleaned(self):
        """Test that token with 'Bearer ' prefix is cleaned."""
        client = Scaleout(host="example.com", token="Bearer some-token-value")
        
        self.mock_token_manager.assert_called_once()
        call_kwargs = self.mock_token_manager.call_args[1]
        # Token should be cleaned (without Bearer prefix)
        self.assertEqual(call_kwargs['refresh_token'], "some-token-value")

    def test_no_token_no_token_manager(self):
        """Test that TokenManager is not created when no token provided."""
        client = Scaleout(host="example.com")
        
        # TokenManager should not be instantiated without token
        self.mock_token_manager.assert_not_called()
        self.assertIsNone(client.token_manager)


class TestScaleoutHeaders(unittest.TestCase):
    """Test cases for header construction."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.cache_patch.stop()

    def test_get_headers_without_token_manager(self):
        """Test _get_headers when no TokenManager is configured."""
        client = Scaleout(host="example.com")
        headers = client._get_headers()
        
        # Should return empty or base headers
        self.assertIsInstance(headers, dict)

    def test_get_headers_with_additional_headers(self):
        """Test _get_headers with additional headers merging."""
        client = Scaleout(host="example.com")
        additional = {"X-Custom": "value", "X-Test": "123"}
        headers = client._get_headers(additional_headers=additional)
        
        self.assertEqual(headers["X-Custom"], "value")
        self.assertEqual(headers["X-Test"], "123")

    def test_get_headers_with_token_manager(self):
        """Test _get_headers with TokenManager configured."""
        with patch('scaleoututil.api.client.TokenManager') as mock_tm_class:
            mock_tm_instance = MagicMock()
            mock_tm_instance.get_auth_header.return_value = {"Authorization": "Bearer test-token"}
            mock_tm_class.return_value = mock_tm_instance
            
            client = Scaleout(host="example.com", token="test-token")
            headers = client._get_headers()
            
            # Should get headers from TokenManager
            self.assertEqual(headers["Authorization"], "Bearer test-token")
            mock_tm_instance.get_auth_header.assert_called_once()


class TestScaleoutTokenManagerFailureFallback(unittest.TestCase):
    """Test that Scaleout gracefully falls back to unauthenticated mode when TokenManager init fails."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_patch = patch('scaleoututil.api.client.TokenCache')
        self.mock_cache = self.cache_patch.start()
        self.mock_cache.return_value.exists.return_value = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.cache_patch.stop()

    def test_runtime_error_falls_back_to_no_auth(self):
        """Test that a RuntimeError from TokenManager results in unauthenticated client."""
        with patch('scaleoututil.api.client.TokenManager', side_effect=RuntimeError("Token refresh failed")):
            client = Scaleout(host="example.com", token="some-stale-token")

        self.assertIsNone(client.token_manager)

    def test_request_exception_falls_back_to_no_auth(self):
        """Test that a network error from TokenManager results in unauthenticated client."""
        with patch(
            'scaleoututil.api.client.TokenManager',
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            client = Scaleout(host="localhost", port=8080, secure=False, token="some-token")

        self.assertIsNone(client.token_manager)

    def test_fallback_get_headers_returns_empty_dict(self):
        """Test that _get_headers returns empty headers (no Authorization) after fallback."""
        with patch('scaleoututil.api.client.TokenManager', side_effect=RuntimeError("No auth endpoint")):
            client = Scaleout(host="example.com", token="some-token")

        headers = client._get_headers()
        self.assertIsInstance(headers, dict)
        self.assertNotIn("Authorization", headers)

    def test_fallback_get_headers_still_merges_additional_headers(self):
        """Test that additional headers are still merged after fallback."""
        with patch('scaleoututil.api.client.TokenManager', side_effect=RuntimeError("No auth endpoint")):
            client = Scaleout(host="example.com", token="some-token")

        headers = client._get_headers(additional_headers={"X-Custom": "value"})
        self.assertEqual(headers["X-Custom"], "value")
        self.assertNotIn("Authorization", headers)

    def test_fallback_logs_warning(self):
        """Test that a warning is logged when falling back to unauthenticated mode."""
        with patch('scaleoututil.api.client.TokenManager', side_effect=RuntimeError("Token refresh failed")):
            with patch('scaleoututil.api.client.ScaleoutLogger') as mock_logger:
                client = Scaleout(host="example.com", token="some-token")

                mock_logger.return_value.warning.assert_called()
                warning_msg = mock_logger.return_value.warning.call_args[0][0]
                self.assertIn("Token authentication failed", warning_msg)
                self.assertIn("no auth system", warning_msg)


if __name__ == "__main__":
    unittest.main()
