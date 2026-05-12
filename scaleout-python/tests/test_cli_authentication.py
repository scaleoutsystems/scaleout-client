"""Unit tests to verify CLI commands requiring authentication call complement_with_context."""

import unittest
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner

# Import all command modules
from scaleout.cli.model_cmd import list_models, get_model, set_active_model
from scaleout.cli.session_cmd import list_sessions, get_session, stop_session, start_session
from scaleout.cli.status_cmd import list_statuses, get_status
from scaleout.cli.validation_cmd import list_validations, get_validation
from scaleout.cli.round_cmd import list_rounds, get_round
from scaleout.cli.combiner_cmd import list_combiners, get_combiner


class TestCLIAuthenticationCalls(unittest.TestCase):
    """Test that CLI commands requiring authentication call complement_with_context."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scaleout.cli.model_cmd.complement_with_context")
    @patch("scaleout.cli.model_cmd.get_response")
    def test_list_models_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_models calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_models, [])

        mock_complement.assert_called_once()
        # Verify it's called with the right parameters (None for all when not provided)
        args = mock_complement.call_args[0]
        self.assertEqual(len(args), 4)  # protocol, host, port, token

    @patch("scaleout.cli.model_cmd.complement_with_context")
    @patch("scaleout.cli.model_cmd.get_response")
    def test_get_model_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_model calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_model, ["--id", "test-id"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.model_cmd.complement_with_context")
    @patch("scaleout.cli.model_cmd.perform_chunked_upload")
    @patch("scaleout.cli.model_cmd.requests.post")
    def test_set_active_model_calls_complement_with_context(self, mock_post, mock_chunked_upload, mock_complement):
        """Test that set_active_model calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_chunked_upload.return_value = "dummy-token"
        mock_post.return_value = MagicMock(status_code=200)

        with self.runner.isolated_filesystem():
            # Create a dummy model file
            with open("model.npz", "w") as f:
                f.write("dummy")

            result = self.runner.invoke(set_active_model, ["--file", "model.npz"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.session_cmd.complement_with_context")
    @patch("scaleout.cli.session_cmd.get_response")
    def test_list_sessions_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_sessions calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_sessions, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.session_cmd.complement_with_context")
    @patch("scaleout.cli.session_cmd.get_response")
    def test_get_session_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_session calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_session, ["--id", "test-id"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.session_cmd.complement_with_context")
    @patch("scaleout.cli.session_cmd.requests.post")
    def test_stop_session_calls_complement_with_context(self, mock_post, mock_complement):
        """Test that stop_session calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"message": "stopped"})

        result = self.runner.invoke(stop_session, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.session_cmd.complement_with_context")
    @patch("scaleout.cli.session_cmd.requests.post")
    def test_start_session_calls_complement_with_context(self, mock_post, mock_complement):
        """Test that start_session calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"session_id": "test-id"})

        result = self.runner.invoke(start_session, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.status_cmd.complement_with_context")
    @patch("scaleout.cli.status_cmd.get_response")
    def test_list_statuses_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_statuses calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_statuses, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.status_cmd.complement_with_context")
    @patch("scaleout.cli.status_cmd.get_response")
    def test_get_status_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_status calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_status, ["--id", "test-id"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.validation_cmd.complement_with_context")
    @patch("scaleout.cli.validation_cmd.get_response")
    def test_list_validations_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_validations calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_validations, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.validation_cmd.complement_with_context")
    @patch("scaleout.cli.validation_cmd.get_response")
    def test_get_validation_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_validation calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_validation, ["--id", "test-id"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.round_cmd.complement_with_context")
    @patch("scaleout.cli.round_cmd.get_response")
    def test_list_rounds_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_rounds calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_rounds, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.round_cmd.complement_with_context")
    @patch("scaleout.cli.round_cmd.get_response")
    def test_get_round_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_round calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_round, ["--id", "test-id"])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.combiner_cmd.complement_with_context")
    @patch("scaleout.cli.combiner_cmd.get_response")
    def test_list_combiners_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that list_combiners calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_combiners, [])

        mock_complement.assert_called_once()

    @patch("scaleout.cli.combiner_cmd.complement_with_context")
    @patch("scaleout.cli.combiner_cmd.get_response")
    def test_get_combiner_calls_complement_with_context(self, mock_get_response, mock_complement):
        """Test that get_combiner calls complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {})

        result = self.runner.invoke(get_combiner, ["--id", "test-id"])

        mock_complement.assert_called_once()


class TestCLIAuthenticationParameters(unittest.TestCase):
    """Test that complement_with_context is called with correct parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scaleout.cli.model_cmd.complement_with_context")
    @patch("scaleout.cli.model_cmd.get_response")
    def test_complement_with_context_receives_cli_parameters(self, mock_get_response, mock_complement):
        """Test that CLI parameters are passed to complement_with_context."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(
            list_models,
            ["--protocol", "https", "--host", "api.example.com", "--port", "8080", "--token", "my-token"],
        )

        # Verify complement_with_context was called with the provided parameters
        mock_complement.assert_called_once_with("https", "api.example.com", "8080", "my-token")

    @patch("scaleout.cli.session_cmd.complement_with_context")
    @patch("scaleout.cli.session_cmd.get_response")
    def test_complement_with_context_receives_none_when_not_provided(self, mock_get_response, mock_complement):
        """Test that None values are passed when CLI parameters are not provided."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_sessions, [])

        # Verify complement_with_context was called with None for all parameters
        mock_complement.assert_called_once_with(None, None, None, None)

    @patch("scaleout.cli.model_cmd.complement_with_context")
    @patch("scaleout.cli.model_cmd.get_response")
    def test_complement_with_context_receives_partial_parameters(self, mock_get_response, mock_complement):
        """Test that mix of provided and None parameters work correctly."""
        mock_complement.return_value = ("http://localhost:8092", "test-token")
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_models, ["--host", "api.example.com", "--token", "my-token"])

        # Verify complement_with_context was called with partial parameters
        mock_complement.assert_called_once_with(None, "api.example.com", None, "my-token")


class TestAuthenticationIntegration(unittest.TestCase):
    """Integration tests to verify authentication flow in CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scaleout.cli.model_cmd.get_response")
    @patch("scaleout.cli.shared.get_all_contexts")
    @patch("scaleout.cli.shared.get_active_context")
    def test_cli_command_uses_context_when_no_host_provided(
        self, mock_get_active, mock_get_contexts, mock_get_response
    ):
        """Test that CLI commands use context data when host is not provided."""
        # Setup context data
        mock_get_active.return_value = 0
        mock_get_contexts.return_value = [
            {"name": "test-context", "host": "http://context-host.com", "token": "context-token"}
        ]
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(list_models, [])

        # Verify the request was made to the context host
        call_kwargs = mock_get_response.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "http://context-host.com")
        self.assertEqual(call_kwargs["token"], "context-token")

    @patch("scaleout.cli.model_cmd.get_response")
    def test_cli_command_uses_provided_host_over_context(self, mock_get_response):
        """Test that explicitly provided host takes precedence over context."""
        mock_get_response.return_value = MagicMock(status_code=200, json=lambda: {"count": 0, "result": []})

        result = self.runner.invoke(
            list_models, ["--host", "http://explicit-host.com", "--token", "explicit-token"]
        )

        # Verify the request was made to the explicit host
        call_kwargs = mock_get_response.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "http://explicit-host.com")
        self.assertEqual(call_kwargs["token"], "explicit-token")


if __name__ == "__main__":
    unittest.main()
