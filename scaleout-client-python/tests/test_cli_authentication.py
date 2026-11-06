"""Unit tests verifying CLI commands build a Scaleout client and delegate to the SDK.

After issue #815 the CLI no longer builds requests itself: every networked command resolves a
:class:`Scaleout` client via ``build_client`` (which bridges the active context + CLI args) and
calls the matching SDK method, rendering the result with ``render_response``.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock, mock_open

from click.testing import CliRunner

from scaleout.cli.shared import build_client, update_active_context_token
from scaleout.cli.model_cmd import list_models, get_model, set_active_model
from scaleout.cli.session_cmd import list_sessions, get_session, stop_session, start_session
from scaleout.cli.status_cmd import list_statuses, get_status
from scaleout.cli.validation_cmd import list_validations, get_validation, start_validation
from scaleout.cli.round_cmd import list_rounds, get_round
from scaleout.cli.combiner_cmd import list_combiners, get_combiner, set_public_hostname
from scaleout.cli.package_cmd import list_packages, get_package, set_active as set_active_package
from scaleout.cli.client_cmd import list_clients, get_client

# (command module, command, CLI args, expected SDK method on the client)
DELEGATION_CASES = [
    ("scaleout.cli.model_cmd", list_models, [], "get_models"),
    ("scaleout.cli.model_cmd", get_model, ["--id", "x"], "get_model"),
    ("scaleout.cli.model_cmd", set_active_model, ["--file", "model.npz"], "set_active_model"),
    ("scaleout.cli.session_cmd", list_sessions, [], "get_sessions"),
    ("scaleout.cli.session_cmd", get_session, ["--id", "x"], "get_session"),
    ("scaleout.cli.session_cmd", start_session, [], "start_session"),
    ("scaleout.cli.status_cmd", list_statuses, [], "get_statuses"),
    ("scaleout.cli.status_cmd", get_status, ["--id", "x"], "get_status"),
    ("scaleout.cli.validation_cmd", list_validations, [], "get_validations"),
    ("scaleout.cli.validation_cmd", get_validation, ["--id", "x"], "get_validation"),
    ("scaleout.cli.validation_cmd", start_validation, ["--session-id", "s", "--model-id", "m"], "start_validation"),
    ("scaleout.cli.round_cmd", list_rounds, [], "get_rounds"),
    ("scaleout.cli.round_cmd", get_round, ["--id", "x"], "get_round"),
    ("scaleout.cli.combiner_cmd", list_combiners, [], "get_combiners"),
    ("scaleout.cli.combiner_cmd", get_combiner, ["--id", "x"], "get_combiner"),
    ("scaleout.cli.package_cmd", list_packages, [], "get_packages"),
    ("scaleout.cli.package_cmd", get_package, ["--id", "x"], "get_package"),
    ("scaleout.cli.package_cmd", set_active_package, ["--file", "p.tgz", "--name", "n"], "set_active_package"),
    ("scaleout.cli.client_cmd", list_clients, [], "get_clients"),
    ("scaleout.cli.client_cmd", get_client, ["--id", "x"], "get_client"),
]


class TestCLIDelegation(unittest.TestCase):
    """Each command builds a client and calls the matching SDK method."""

    def setUp(self):
        self.runner = CliRunner()

    def test_commands_build_client_and_delegate(self):
        for module, command, args, method in DELEGATION_CASES:
            with self.subTest(command=command.name):
                with ExitStack() as stack:
                    mock_build = stack.enter_context(patch(f"{module}.build_client"))
                    mock_render = stack.enter_context(patch(f"{module}.render_response"))
                    mock_client = MagicMock()
                    mock_build.return_value = ("http://localhost:8092", mock_client)
                    mock_render.return_value = 0

                    result = self.runner.invoke(command, args)

                    self.assertEqual(result.exit_code, 0, msg=result.output)
                    mock_build.assert_called_once()
                    getattr(mock_client, method).assert_called_once()
                    mock_render.assert_called_once()

    def test_reporting_commands_delegate_and_report(self):
        """Mutation commands that return a {"message": ...} payload delegate to the SDK and print the message.

        These report success/failure explicitly (build_client + try/except) instead of via render_response.
        """
        # (module, command, args, SDK method, expected call args)
        reporting_cases = [
            ("scaleout.cli.combiner_cmd", set_public_hostname, ["--id", "x", "myhost"], "set_public_hostname", ("x", "myhost")),
            ("scaleout.cli.session_cmd", stop_session, [], "stop_current_command", ()),
        ]
        for module, command, args, method, expected_args in reporting_cases:
            with self.subTest(command=command.name):
                with patch(f"{module}.build_client") as mock_build:
                    mock_client = MagicMock()
                    getattr(mock_client, method).return_value = {"message": "done ok"}
                    mock_build.return_value = ("http://localhost:8092", mock_client)

                    result = self.runner.invoke(command, args)

                    self.assertEqual(result.exit_code, 0, msg=result.output)
                    mock_build.assert_called_once()
                    getattr(mock_client, method).assert_called_once_with(*expected_args)
                    self.assertIn("done ok", result.output)


class TestBuildClientArguments(unittest.TestCase):
    """build_client receives CLI parameters (including --no-verify-tls) verbatim."""

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_parameters_forwarded(self):
        with patch("scaleout.cli.model_cmd.build_client") as mock_build, patch("scaleout.cli.model_cmd.render_response") as mock_render:
            mock_build.return_value = ("http://api.example.com:8080", MagicMock())
            mock_render.return_value = 0

            result = self.runner.invoke(
                list_models,
                ["--protocol", "https", "--host", "api.example.com", "--port", "8080", "--token", "my-token"],
            )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            mock_build.assert_called_once_with("https", "api.example.com", "8080", "my-token", False)

    def test_no_verify_tls_flag_forwarded(self):
        with patch("scaleout.cli.model_cmd.build_client") as mock_build, patch("scaleout.cli.model_cmd.render_response") as mock_render:
            mock_build.return_value = ("http://localhost:8092", MagicMock())
            mock_render.return_value = 0

            result = self.runner.invoke(list_models, ["--no-verify-tls"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            mock_build.assert_called_once_with(None, None, None, None, True)


class TestBuildClient(unittest.TestCase):
    """build_client resolves host/token from the active context and constructs Scaleout via Login."""

    @patch("scaleout.cli.shared.Login")
    @patch("scaleout.cli.shared.Scaleout")
    @patch("scaleout.cli.shared.get_all_contexts")
    @patch("scaleout.cli.shared.get_active_context")
    def test_uses_active_context(self, mock_active, mock_contexts, mock_scaleout, mock_login):
        mock_active.return_value = 0
        mock_contexts.return_value = [{"name": "c", "host": "http://context-host.com", "token": "context-token"}]
        mock_login.return_value.get_access_token.return_value = "fresh-access-token"

        base_url, _ = build_client(None, None, None, None)

        self.assertEqual(base_url, "http://context-host.com")
        # The context token is used as the refresh credential for Login (not sent as a bearer).
        _, login_kwargs = mock_login.call_args
        self.assertEqual(login_kwargs["server_url"], "http://context-host.com")
        self.assertEqual(login_kwargs["credential"], "context-token")
        self.assertTrue(login_kwargs["verify_ssl"])
        # Credential came from the active context -> a rotation callback is wired up.
        self.assertIsNotNone(login_kwargs["on_token_refresh"])

        _, kwargs = mock_scaleout.call_args
        self.assertEqual(kwargs["host"], "http://context-host.com")
        self.assertTrue(kwargs["verify"])
        # The provider returns Login's (refreshed) access token, not the raw refresh token.
        self.assertEqual(kwargs["access_token_provider"](), "fresh-access-token")
        self.assertNotIn("token", kwargs)

    @patch("scaleout.cli.shared.Login")
    @patch("scaleout.cli.shared.Scaleout")
    def test_explicit_host_and_no_verify_tls(self, mock_scaleout, mock_login):
        mock_login.return_value.get_access_token.return_value = "fresh-access-token"

        base_url, _ = build_client(None, "http://explicit-host.com", None, "explicit-token", no_verify_tls=True)

        self.assertEqual(base_url, "http://explicit-host.com")
        _, login_kwargs = mock_login.call_args
        self.assertEqual(login_kwargs["credential"], "explicit-token")
        self.assertFalse(login_kwargs["verify_ssl"])
        # Explicit --token/--host -> no write-back to any context.
        self.assertIsNone(login_kwargs["on_token_refresh"])

        _, kwargs = mock_scaleout.call_args
        self.assertEqual(kwargs["host"], "http://explicit-host.com")
        self.assertFalse(kwargs["verify"])
        self.assertEqual(kwargs["access_token_provider"](), "fresh-access-token")

    @patch("scaleout.cli.shared.Login")
    @patch("scaleout.cli.shared.Scaleout")
    def test_access_token_provider_returns_none_on_auth_failure(self, mock_scaleout, mock_login):
        """If the refresh credential can't produce an access token, the provider returns None
        (request proceeds unauthenticated -> clean error) rather than raising."""
        mock_login.return_value.get_access_token.side_effect = RuntimeError("no credential")

        build_client(None, "http://h", None, "tok")

        _, kwargs = mock_scaleout.call_args
        self.assertIsNone(kwargs["access_token_provider"]())

    @patch("scaleout.cli.shared.update_active_context_token")
    @patch("scaleout.cli.shared.Login")
    @patch("scaleout.cli.shared.Scaleout")
    @patch("scaleout.cli.shared.get_all_contexts")
    @patch("scaleout.cli.shared.get_active_context")
    def test_rotated_refresh_token_persisted_to_context(self, mock_active, mock_contexts, mock_scaleout, mock_login, mock_update):
        mock_active.return_value = 0
        mock_contexts.return_value = [{"name": "c", "host": "http://h", "token": "rt0"}]

        build_client(None, None, None, None)

        _, login_kwargs = mock_login.call_args
        callback = login_kwargs["on_token_refresh"]
        self.assertIsNotNone(callback)
        # Simulate Hydra rotating the refresh token during a refresh.
        callback("new-access", "rt1", None)
        mock_update.assert_called_once_with("rt1")


class TestUpdateActiveContextToken(unittest.TestCase):
    """update_active_context_token rewrites only the active context's token in contexts.yaml."""

    @patch("scaleout.cli.shared.get_all_contexts")
    @patch("scaleout.cli.shared.get_active_context")
    def test_writes_new_token_for_active_context(self, mock_active, mock_contexts):
        mock_active.return_value = 1
        contexts = [
            {"name": "a", "host": "http://a", "token": "a-tok"},
            {"name": "b", "host": "http://b", "token": "old"},
        ]
        mock_contexts.return_value = contexts

        with patch("builtins.open", mock_open()) as m, patch("scaleout.cli.shared.yaml.dump") as mock_dump:
            update_active_context_token("new")

        m.assert_called_once()
        written = mock_dump.call_args[0][0]
        self.assertEqual(written[1]["token"], "new")  # active context updated
        self.assertEqual(written[0]["token"], "a-tok")  # others untouched

    @patch("scaleout.cli.shared.get_active_context", return_value=None)
    def test_no_active_context_is_noop(self, _mock_active):
        with patch("builtins.open", mock_open()) as m:
            update_active_context_token("new")
        m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
