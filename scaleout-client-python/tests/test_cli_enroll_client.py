"""Unit tests for `scaleout client enroll`.

Regression coverage for a bug where an enrollment failure (e.g. an invalid/expired
enrollment token) was swallowed and the command exited 0 anyway, which let CI's
`subprocess.run([...], check=True)` calls sail past a broken enrollment.
"""

import unittest
from unittest.mock import patch

from click.testing import CliRunner

from scaleout.cli.client_cmd import enroll_client


class TestEnrollClient(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('scaleout.cli.client_cmd._enroll_client')
    def test_enrollment_failure_exits_nonzero(self, mock_enroll):
        mock_enroll.side_effect = RuntimeError("Enrollment token is invalid or expired")

        result = self.runner.invoke(enroll_client, ["--enrollment-token", "bad-token", "-u", "http://localhost:8092", "--no-config"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Enrollment failed", result.output)

    @patch('scaleout.cli.client_cmd.TokenCache')
    @patch('scaleout.cli.client_cmd._enroll_client')
    def test_enrollment_success_exits_zero(self, mock_enroll, mock_cache_class):
        mock_enroll.return_value = ("client-123", "access-tok", "refresh-tok")

        result = self.runner.invoke(enroll_client, ["--enrollment-token", "good-token", "-u", "http://localhost:8092", "--no-config"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(result.stdout.strip(), "client-123")


if __name__ == '__main__':
    unittest.main()
