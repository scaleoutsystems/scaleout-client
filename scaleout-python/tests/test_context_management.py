"""Unit tests for context management functions in shared.py."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yaml

from scaleout.cli.shared import (
    set_context,
    set_active_context,
    get_active_context,
    get_all_contexts,
    get_context_by_index,
    switch_context,
    remove_context,
    complement_with_context,
    HOME_DIR,
    CONTEXT_FOLDER,
)


class TestContextManagement(unittest.TestCase):
    """Test cases for context management functions."""

    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.contexts_file = os.path.join(self.temp_dir, CONTEXT_FOLDER, "contexts.yaml")
        self.active_file = os.path.join(self.temp_dir, CONTEXT_FOLDER, "active.yaml")
        self.context_dir = os.path.join(self.temp_dir, CONTEXT_FOLDER)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_creates_directory(self, mock_home_dir):
        """Test that set_context creates the context directory if it doesn't exist."""
        mock_home_dir.__str__ = lambda _: self.temp_dir
        mock_home_dir.__add__ = lambda self, other: os.path.join(self.temp_dir, other.lstrip("/"))

        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://localhost:8092", "test-token", "test-context")

        self.assertTrue(os.path.exists(self.context_dir))

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_with_name(self, mock_home_dir):
        """Test set_context saves context with provided name."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://localhost:8092", "test-token", "my-context")

            with open(self.contexts_file, "r") as f:
                contexts = yaml.safe_load(f)

            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0]["name"], "my-context")
            self.assertEqual(contexts[0]["host"], "http://localhost:8092")
            self.assertEqual(contexts[0]["token"], "test-token")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_without_name_uses_host(self, mock_home_dir):
        """Test set_context uses host as name when no name is provided."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://api.example.com", "test-token")

            with open(self.contexts_file, "r") as f:
                contexts = yaml.safe_load(f)

            self.assertEqual(contexts[0]["name"], "api.example.com")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_appends_to_existing_contexts(self, mock_home_dir):
        """Test set_context appends new context to existing contexts."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "context1")
            set_context("http://host2.com", "token2", "context2")

            with open(self.contexts_file, "r") as f:
                contexts = yaml.safe_load(f)

            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[0]["name"], "context1")
            self.assertEqual(contexts[1]["name"], "context2")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_allows_duplicate_names_different_hosts(self, mock_home_dir):
        """Test that multiple contexts can share the same name if they have different hosts."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "production")
            set_context("http://host2.com", "token2", "production")

            with open(self.contexts_file, "r") as f:
                contexts = yaml.safe_load(f)

            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[0]["name"], "production")
            self.assertEqual(contexts[1]["name"], "production")
            self.assertEqual(contexts[0]["host"], "http://host1.com")
            self.assertEqual(contexts[1]["host"], "http://host2.com")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_context_updates_identical_host_and_name(self, mock_home_dir):
        """Test that setting a context with identical host and name updates instead of appending."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "production")
            set_context("http://host1.com", "token2", "production")

            with open(self.contexts_file, "r") as f:
                contexts = yaml.safe_load(f)

            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0]["token"], "token2")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_set_active_context(self, mock_home_dir):
        """Test set_active_context saves the active index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            set_active_context(2)

            with open(self.active_file, "r") as f:
                data = yaml.safe_load(f)

            self.assertEqual(data["active_index"], 2)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_active_context_returns_index(self, mock_home_dir):
        """Test get_active_context returns the correct index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            with open(self.active_file, "w") as f:
                yaml.dump({"active_index": 3}, f)

            index = get_active_context()

            self.assertEqual(index, 3)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_active_context_returns_none_when_no_file(self, mock_home_dir):
        """Test get_active_context returns None when no active file exists."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            index = get_active_context()
            self.assertIsNone(index)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_active_context_handles_old_format(self, mock_home_dir):
        """Test get_active_context returns None for old name-based format."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            with open(self.active_file, "w") as f:
                yaml.dump({"active": "old-context-name"}, f)

            index = get_active_context()

            self.assertIsNone(index)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_all_contexts_returns_list(self, mock_home_dir):
        """Test get_all_contexts returns a list of contexts."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            contexts_data = [
                {"name": "ctx1", "host": "http://host1.com", "token": "token1"},
                {"name": "ctx2", "host": "http://host2.com", "token": "token2"},
            ]
            with open(self.contexts_file, "w") as f:
                yaml.dump(contexts_data, f)

            contexts = get_all_contexts()

            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[0]["name"], "ctx1")
            self.assertEqual(contexts[1]["name"], "ctx2")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_all_contexts_returns_empty_list_when_no_file(self, mock_home_dir):
        """Test get_all_contexts returns empty list when no file exists."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            contexts = get_all_contexts()
            self.assertEqual(contexts, [])

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_all_contexts_migrates_old_dict_format(self, mock_home_dir):
        """Test get_all_contexts converts old dict format to list."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            old_format = {
                "context1": {"host": "http://host1.com", "token": "token1"},
                "context2": {"host": "http://host2.com", "token": "token2"},
            }
            with open(self.contexts_file, "w") as f:
                yaml.dump(old_format, f)

            contexts = get_all_contexts()

            self.assertIsInstance(contexts, list)
            self.assertEqual(len(contexts), 2)
            # Check that names are added
            names = [c["name"] for c in contexts]
            self.assertIn("context1", names)
            self.assertIn("context2", names)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_context_by_index_returns_correct_context(self, mock_home_dir):
        """Test get_context_by_index returns the context at specified index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")

            context = get_context_by_index(1)

            self.assertEqual(context["name"], "ctx2")
            self.assertEqual(context["host"], "http://host2.com")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_get_context_by_index_returns_none_for_invalid_index(self, mock_home_dir):
        """Test get_context_by_index returns None for invalid indices."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")

            self.assertIsNone(get_context_by_index(-1))
            self.assertIsNone(get_context_by_index(5))

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_switch_context_updates_active_index(self, mock_home_dir):
        """Test switch_context updates the active index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")

            result = switch_context(0)

            self.assertTrue(result)
            self.assertEqual(get_active_context(), 0)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_switch_context_returns_false_for_invalid_index(self, mock_home_dir):
        """Test switch_context returns False for invalid index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")

            result = switch_context(10)

            self.assertFalse(result)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_remove_context_removes_by_index(self, mock_home_dir):
        """Test remove_context removes the context at specified index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")
            set_context("http://host3.com", "token3", "ctx3")

            result = remove_context(1)

            self.assertTrue(result)
            contexts = get_all_contexts()
            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[0]["name"], "ctx1")
            self.assertEqual(contexts[1]["name"], "ctx3")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_remove_context_switches_to_first_when_removing_active(self, mock_home_dir):
        """Test remove_context switches to index 0 when removing active context."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")
            set_context("http://host3.com", "token3", "ctx3")
            # Last set_context sets active to index 2

            result = remove_context(2)

            self.assertTrue(result)
            # Should switch to index 0
            self.assertEqual(get_active_context(), 0)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_remove_context_adjusts_active_index_when_removing_before_active(self, mock_home_dir):
        """Test remove_context adjusts active index when removing a context before it."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")
            set_context("http://host3.com", "token3", "ctx3")
            # Active is at index 2

            # Remove index 0, active should shift from 2 to 1
            result = remove_context(0)

            self.assertTrue(result)
            self.assertEqual(get_active_context(), 1)

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_remove_context_clears_active_when_removing_last_context(self, mock_home_dir):
        """Test remove_context clears active file when removing the last context."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")

            result = remove_context(0)

            self.assertTrue(result)
            self.assertFalse(os.path.exists(self.active_file))
            self.assertIsNone(get_active_context())

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_remove_context_returns_false_for_invalid_index(self, mock_home_dir):
        """Test remove_context returns False for invalid index."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")

            result = remove_context(5)

            self.assertFalse(result)

    @patch("scaleout.cli.shared.HOME_DIR")
    @patch.dict(os.environ, {}, clear=True)
    def test_complement_with_context_uses_active_context(self, mock_home_dir):
        """Test complement_with_context retrieves host from active context."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            set_context("http://host1.com", "token1", "ctx1")
            set_context("http://host2.com", "token2", "ctx2")
            switch_context(0)

            host, token = complement_with_context(None, None, None, None)

            self.assertEqual(host, "http://host1.com")
            self.assertEqual(token, "token1")

    @patch("scaleout.cli.shared.HOME_DIR")
    @patch.dict(os.environ, {}, clear=True)
    def test_complement_with_context_uses_first_when_no_active(self, mock_home_dir):
        """Test complement_with_context uses first context when no active context."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            # Manually create contexts without setting active
            os.makedirs(self.context_dir, exist_ok=True)
            contexts = [
                {"name": "ctx1", "host": "http://host1.com", "token": "token1"},
                {"name": "ctx2", "host": "http://host2.com", "token": "token2"},
            ]
            with open(self.contexts_file, "w") as f:
                yaml.dump(contexts, f)

            host, token = complement_with_context(None, None, None, None)

            self.assertEqual(host, "http://host1.com")
            self.assertEqual(token, "token1")

    @patch("scaleout.cli.shared.HOME_DIR")
    @patch.dict(os.environ, {"SCALEOUT_AUTH_TOKEN": "env-token"}, clear=True)
    def test_complement_with_context_uses_env_token_when_no_context_token(self, mock_home_dir):
        """Test complement_with_context uses environment token as fallback."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            contexts = [{"name": "ctx1", "host": "http://host1.com"}]  # No token
            with open(self.contexts_file, "w") as f:
                yaml.dump(contexts, f)
            set_active_context(0)

            host, token = complement_with_context(None, None, None, None)

            self.assertEqual(token, "env-token")

    @patch("scaleout.cli.shared.HOME_DIR")
    @patch.dict(os.environ, {}, clear=True)
    def test_complement_with_context_returns_provided_host(self, mock_home_dir):
        """Test complement_with_context returns provided host without context lookup."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            host, token = complement_with_context(None, "http://custom.com", None, "custom-token")

            self.assertEqual(host, "http://custom.com")
            self.assertEqual(token, "custom-token")

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_complement_with_context_raises_error_when_protocol_without_host(self, mock_home_dir):
        """Test complement_with_context raises error when protocol is provided without host."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            with self.assertRaises(ValueError) as context:
                complement_with_context("https", None, None, None)

            self.assertIn("Both protocol and port must be provided together with host", str(context.exception))

    @patch("scaleout.cli.shared.HOME_DIR")
    def test_complement_with_context_raises_error_when_port_without_host(self, mock_home_dir):
        """Test complement_with_context raises error when port is provided without host."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            with self.assertRaises(ValueError) as context:
                complement_with_context(None, None, "8092", None)

            self.assertIn("Both protocol and port must be provided together with host", str(context.exception))

    @patch("scaleout.cli.shared.HOME_DIR")
    @patch.dict(os.environ, {}, clear=True)
    def test_complement_with_context_returns_none_when_no_contexts(self, mock_home_dir):
        """Test complement_with_context returns None for host when no contexts exist."""
        with patch("scaleout.cli.shared.HOME_DIR", self.temp_dir):
            host, token = complement_with_context(None, None, None, None)

            self.assertIsNone(host)
            self.assertIsNone(token)


if __name__ == "__main__":
    unittest.main()
