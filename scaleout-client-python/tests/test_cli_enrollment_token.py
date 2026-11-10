"""Unit tests for `scaleout client create-enrollment-token`.

Credential resolution goes through `build_client` (so -H/-t or the active context
work the same as every other command), then the command posts directly to
/api/v1/auth/enrollment-tokens with the resolved headers.
"""

import unittest
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from scaleout.cli.client_cmd import create_enrollment_token


class TestCreateEnrollmentToken(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('scaleout.cli.client_cmd.requests.post')
    @patch('scaleout.cli.client_cmd.build_client')
    def test_success_prints_token_to_stdout(self, mock_build_client, mock_post):
        mock_client = MagicMock()
        mock_client._get_headers.return_value = {"Authorization": "Bearer abc"}
        mock_client.verify = True
        mock_build_client.return_value = ("http://localhost:8092", mock_client)

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"token": "enroll-jwt", "expires_at": "2026-01-01T00:00:00Z"})

        result = self.runner.invoke(create_enrollment_token, ["--name", "my-token"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(result.stdout.strip(), "enroll-jwt")
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        self.assertTrue(called_url.endswith("/api/v1/auth/enrollment-tokens"))
        self.assertEqual(mock_post.call_args.kwargs["headers"], {"Authorization": "Bearer abc"})

    @patch('scaleout.cli.client_cmd.requests.post')
    @patch('scaleout.cli.client_cmd.build_client')
    def test_no_authorization_header_fails_without_network_call(self, mock_build_client, mock_post):
        mock_client = MagicMock()
        mock_client._get_headers.return_value = {}
        mock_build_client.return_value = ("http://localhost:8092", mock_client)

        result = self.runner.invoke(create_enrollment_token, ["--name", "my-token"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No token found", result.output)
        mock_post.assert_not_called()

    @patch('scaleout.cli.client_cmd.build_client')
    def test_build_client_failure_reports_error_and_exits_nonzero(self, mock_build_client):
        mock_build_client.side_effect = RuntimeError("no active context")

        result = self.runner.invoke(create_enrollment_token, ["--name", "my-token"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Failed to resolve host/credentials", result.output)

    @patch('scaleout.cli.client_cmd.requests.post')
    @patch('scaleout.cli.client_cmd.build_client')
    def test_unauthorized_response_exits_nonzero(self, mock_build_client, mock_post):
        mock_client = MagicMock()
        mock_client._get_headers.return_value = {"Authorization": "Bearer expired"}
        mock_client.verify = True
        mock_build_client.return_value = ("http://localhost:8092", mock_client)
        mock_post.return_value = MagicMock(status_code=401)

        result = self.runner.invoke(create_enrollment_token, ["--name", "my-token"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unauthorized", result.output)


if __name__ == '__main__':
    unittest.main()
