"""Unit tests for SDK additions/fixes made in issue #815.

Covers: set_public_hostname, start_validation, the set_active_model unsupported-extension guard,
and start_session helper auto-detection.
"""

import unittest
from unittest.mock import patch, MagicMock

from scaleoututil.api.client import Scaleout


class TestScaleout815(unittest.TestCase):
    def setUp(self):
        self.token_patch = patch("scaleoututil.api.client.Login")
        self.token_patch.start()
        self.cache_patch = patch("scaleoututil.api.client.TokenCache")
        mock_cache = self.cache_patch.start()
        mock_cache.return_value.exists.return_value = False
        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        self.token_patch.stop()
        self.cache_patch.stop()

    @patch("requests.patch")
    def test_set_public_hostname(self, mock_patch):
        mock_patch.return_value = MagicMock(status_code=200, json=lambda: {"message": "Combiner updated"})

        result = self.client.set_public_hostname("c1", "host.example")

        self.assertEqual(result, {"message": "Combiner updated"})
        url = mock_patch.call_args[0][0]
        self.assertTrue(url.endswith("/api/v1/combiners/c1"))
        self.assertEqual(mock_patch.call_args[1]["json"], {"public_hostname": "host.example"})

    @patch("requests.patch")
    def test_set_public_hostname_raises_on_error(self, mock_patch):
        """A non-success status raises RuntimeError carrying the server message so the CLI can report failure and exit non-zero."""
        mock_patch.return_value = MagicMock(status_code=404, json=lambda: {"message": "Entity with id: c1 not found"})

        with self.assertRaises(RuntimeError) as cm:
            self.client.set_public_hostname("c1", "host.example")
        self.assertEqual(str(cm.exception), "Entity with id: c1 not found")

    @patch("requests.post")
    def test_stop_current_command_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"message": "stopped"})

        result = self.client.stop_current_command()

        self.assertEqual(result, {"message": "stopped"})
        self.assertTrue(mock_post.call_args[0][0].endswith("/api/v1/control/stop"))

    @patch("requests.post")
    def test_stop_current_command_raises_on_error(self, mock_post):
        """A non-success status raises RuntimeError so the CLI reports failure and exits non-zero."""
        mock_post.return_value = MagicMock(status_code=500, json=lambda: {"message": "No active session"})

        with self.assertRaises(RuntimeError) as cm:
            self.client.stop_current_command()
        self.assertEqual(str(cm.exception), "No active session")

    @patch("requests.post")
    def test_start_validation(self, mock_post):
        mock_post.return_value = MagicMock(json=lambda: {"message": "started"})

        result = self.client.start_validation("sess1", "model1")

        self.assertEqual(result, {"message": "started"})
        url = mock_post.call_args[0][0]
        self.assertTrue(url.endswith("/api/v1/validations/start"))
        self.assertEqual(mock_post.call_args[1]["json"], {"session_id": "sess1", "model_id": "model1"})

    def test_set_active_model_unsupported_extension(self):
        """An unsupported file type raises RuntimeError so the CLI can report a clear error and exit non-zero."""
        with self.assertRaises(RuntimeError) as cm:
            self.client.set_active_model("/tmp/model.txt")
        self.assertEqual(str(cm.exception), "Unsupported file type. Only .npz and .bin files are supported.")

    @patch("requests.post")
    @patch("requests.get")
    def test_start_session_auto_detects_helper(self, mock_get, mock_post):
        # helpers/active returns a helper name
        mock_get.return_value = MagicMock(status_code=200, json=lambda: "myhelper")
        # sessions/ -> 201, then sessions/start -> 200
        mock_post.side_effect = [
            MagicMock(status_code=201, json=lambda: {"session_id": "sess1"}),
            MagicMock(status_code=200, json=lambda: {"ok": True}),
        ]

        result = self.client.start_session(model_id="m1")  # helper defaults to None -> auto-detect

        self.assertEqual(result["session_id"], "sess1")
        # The active helper should have been propagated into the session config.
        session_post = mock_post.call_args_list[0]
        self.assertEqual(session_post[1]["json"]["session_config"]["helper_type"], "myhelper")
        get_url = mock_get.call_args[0][0]
        self.assertTrue(get_url.endswith("/api/v1/helpers/active"))


if __name__ == "__main__":
    unittest.main()
