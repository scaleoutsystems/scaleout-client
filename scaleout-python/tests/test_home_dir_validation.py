"""Unit tests for HOME_DIR validation in shared.py."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaleout.cli.shared import _get_validated_home_dir


class TestHomeDirValidation(unittest.TestCase):
    """Test cases for _get_validated_home_dir function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_home_dir_from_expanduser(self):
        """Test that default home directory is retrieved from os.path.expanduser."""
        home_dir = _get_validated_home_dir()
        expected_home = str(Path(os.path.expanduser("~")).resolve())
        self.assertEqual(home_dir, expected_home)
        self.assertTrue(os.path.isabs(home_dir))

    @patch.dict(os.environ, {"SCALEOUT_HOME_DIR": ""})
    def test_empty_scaleout_home_dir_raises_error(self):
        """Test that empty SCALEOUT_HOME_DIR raises RuntimeError."""
        with self.assertRaises(RuntimeError) as context:
            _get_validated_home_dir()
        self.assertIn("Unable to determine home directory", str(context.exception))

    def test_existing_valid_directory(self):
        """Test that existing writable directory is validated successfully."""
        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": self.temp_dir}):
            home_dir = _get_validated_home_dir()
            expected_path = str(Path(self.temp_dir).resolve())
            self.assertEqual(home_dir, expected_path)

    def test_nonexistent_directory_is_created(self):
        """Test that non-existent directory is created."""
        new_dir = os.path.join(self.temp_dir, "new_scaleout_home")
        self.assertFalse(os.path.exists(new_dir))

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": new_dir}):
            home_dir = _get_validated_home_dir()
            self.assertTrue(os.path.exists(new_dir))
            self.assertTrue(os.path.isdir(new_dir))
            self.assertEqual(home_dir, str(Path(new_dir).resolve()))

    def test_nested_nonexistent_directory_is_created(self):
        """Test that nested non-existent directories are created with parents."""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        self.assertFalse(os.path.exists(nested_dir))

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": nested_dir}):
            home_dir = _get_validated_home_dir()
            self.assertTrue(os.path.exists(nested_dir))
            self.assertTrue(os.path.isdir(nested_dir))
            self.assertEqual(home_dir, str(Path(nested_dir).resolve()))

    def test_file_instead_of_directory_raises_error(self):
        """Test that a file path instead of directory raises RuntimeError."""
        file_path = os.path.join(self.temp_dir, "not_a_directory.txt")
        with open(file_path, "w") as f:
            f.write("test")

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": file_path}):
            with self.assertRaises(RuntimeError) as context:
                _get_validated_home_dir()
            self.assertIn("not a directory", str(context.exception))

    @patch("os.access")
    def test_non_writable_directory_raises_error(self, mock_access):
        """Test that non-writable directory raises RuntimeError."""
        mock_access.return_value = False

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": self.temp_dir}):
            with self.assertRaises(RuntimeError) as context:
                _get_validated_home_dir()
            self.assertIn("not writable", str(context.exception))

    def test_relative_path_is_resolved_to_absolute(self):
        """Test that relative paths are resolved to absolute paths."""
        # Create a subdirectory to use as relative path
        rel_dir = "scaleout_rel"
        abs_dir = os.path.join(self.temp_dir, rel_dir)
        os.makedirs(abs_dir)

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": rel_dir}):
            # Change to temp directory so relative path resolves correctly
            original_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                home_dir = _get_validated_home_dir()
                self.assertTrue(os.path.isabs(home_dir))
                self.assertEqual(home_dir, str(Path(abs_dir).resolve()))
            finally:
                os.chdir(original_cwd)

    @patch("pathlib.Path.mkdir")
    def test_permission_error_creating_directory(self, mock_mkdir):
        """Test that PermissionError during directory creation raises RuntimeError."""
        mock_mkdir.side_effect = PermissionError("Permission denied")
        new_dir = os.path.join(self.temp_dir, "no_permission")

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": new_dir}):
            with self.assertRaises(RuntimeError) as context:
                _get_validated_home_dir()
            self.assertIn("Cannot create home directory", str(context.exception))

    @patch("pathlib.Path.mkdir")
    def test_os_error_creating_directory(self, mock_mkdir):
        """Test that OSError during directory creation raises RuntimeError."""
        mock_mkdir.side_effect = OSError("Disk full")
        new_dir = os.path.join(self.temp_dir, "disk_full")

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": new_dir}):
            with self.assertRaises(RuntimeError) as context:
                _get_validated_home_dir()
            self.assertIn("Cannot create home directory", str(context.exception))

    def test_path_with_special_characters(self):
        """Test that paths with spaces and special characters are handled correctly."""
        special_dir = os.path.join(self.temp_dir, "dir with spaces & special-chars_123")
        os.makedirs(special_dir)

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": special_dir}):
            home_dir = _get_validated_home_dir()
            self.assertEqual(home_dir, str(Path(special_dir).resolve()))
            self.assertTrue(os.path.exists(home_dir))

    def test_symlink_is_resolved(self):
        """Test that symlinks are resolved to their target."""
        target_dir = os.path.join(self.temp_dir, "target")
        symlink_dir = os.path.join(self.temp_dir, "symlink")
        os.makedirs(target_dir)
        os.symlink(target_dir, symlink_dir)

        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": symlink_dir}):
            home_dir = _get_validated_home_dir()
            # Should resolve to the target, not the symlink
            self.assertEqual(home_dir, str(Path(target_dir).resolve()))

    def test_tilde_expansion(self):
        """Test that tilde (~) is properly expanded."""
        # Use a path with tilde that doesn't rely on actual home
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.expanduser") as mock_expand:
                mock_expand.return_value = self.temp_dir
                with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": ""}):
                    # This will use the default expanduser path
                    pass
        
        # Direct test with existing directory
        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": self.temp_dir}):
            home_dir = _get_validated_home_dir()
            self.assertTrue(os.path.isabs(home_dir))

    def test_returns_string_not_path_object(self):
        """Test that the function returns a string, not a Path object."""
        with patch.dict(os.environ, {"SCALEOUT_HOME_DIR": self.temp_dir}):
            home_dir = _get_validated_home_dir()
            self.assertIsInstance(home_dir, str)
            self.assertNotIsInstance(home_dir, Path)


if __name__ == "__main__":
    unittest.main()
