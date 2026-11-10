#!/usr/bin/env python3
"""Unit tests for ScaleoutLogger behavior."""

import io
import logging
import os
import sys
import unittest
from unittest.mock import patch

from scaleoututil.logging import ScaleoutLogger


class TestScaleoutLogger(unittest.TestCase):
    """Test cases for ScaleoutLogger."""

    def setUp(self):
        """Reset singleton and environment before each test."""
        ScaleoutLogger._instance = None
        # Clean up environment variables
        os.environ.pop("SCALEOUT_LOG_CONSOLE", None)
        os.environ.pop("SCALEOUT_LOG_LEVEL", None)
        # Clear any existing handlers from the "scaleout" logger
        scaleout_logger = logging.getLogger("scaleout")
        scaleout_logger.handlers.clear()
        scaleout_logger.setLevel(logging.NOTSET)

    def tearDown(self):
        """Clean up after each test."""
        ScaleoutLogger._instance = None
        os.environ.pop("SCALEOUT_LOG_CONSOLE", None)
        os.environ.pop("SCALEOUT_LOG_LEVEL", None)

    def test_library_usage_defaults_to_warning(self):
        """Test that library usage without configuration defaults to WARNING level."""
        logger = ScaleoutLogger()
        # Check that the logger level is WARNING
        self.assertEqual(logger.logger.level, logging.WARNING)
        
        # Check that a NullHandler was added (library best practice)
        handler_types = [type(h).__name__ for h in logger.logger.handlers]
        self.assertIn('NullHandler', handler_types)

    def test_enable_console_logging(self):
        """Test that enable_console_logging adds StreamHandler and sets level."""
        logger = ScaleoutLogger()
        
        # Initially should have NullHandler
        initial_handlers = [type(h).__name__ for h in logger.logger.handlers]
        self.assertIn('NullHandler', initial_handlers)
        
        # Enable console logging
        logger.enable_console_logging(logging.DEBUG)
        
        # Should now have StreamHandler and no NullHandler
        final_handlers = [type(h).__name__ for h in logger.logger.handlers]
        self.assertIn('StreamHandler', final_handlers)
        self.assertNotIn('NullHandler', final_handlers)
        
        # Level should be DEBUG
        self.assertEqual(logger.logger.level, logging.DEBUG)

    def test_environment_variable_log_level(self):
        """Test that SCALEOUT_LOG_LEVEL environment variable sets log level."""
        os.environ["SCALEOUT_LOG_LEVEL"] = "DEBUG"
        os.environ["SCALEOUT_LOG_CONSOLE"] = "true"
        
        logger = ScaleoutLogger()
        
        # Should be DEBUG level
        self.assertEqual(logger.logger.level, logging.DEBUG)
        
        # Should have StreamHandler (console enabled)
        handler_types = [type(h).__name__ for h in logger.logger.handlers]
        self.assertIn('StreamHandler', handler_types)

    def test_invalid_log_level_fallback(self):
        """Test that invalid log level falls back to WARNING with a warning message."""
        os.environ["SCALEOUT_LOG_LEVEL"] = "INVALID_LEVEL"
        os.environ["SCALEOUT_LOG_CONSOLE"] = "true"
        
        # Capture log output
        with patch.object(logging.Logger, 'warning') as mock_warning:
            logger = ScaleoutLogger()
            
            # Should have logged a warning about invalid level
            mock_warning.assert_called()
            warning_msg = mock_warning.call_args[0][0]
            self.assertIn("Invalid log level", warning_msg)
            self.assertIn("INVALID_LEVEL", warning_msg)
        
        # Should fall back to WARNING level
        self.assertEqual(logger.logger.level, logging.WARNING)

    def test_singleton_behavior(self):
        """Test that ScaleoutLogger is a singleton."""
        logger1 = ScaleoutLogger()
        logger2 = ScaleoutLogger()
        
        # Should be the same instance
        self.assertIs(logger1, logger2)

    def test_set_log_level_from_string(self):
        """Test setting log level from string."""
        logger = ScaleoutLogger()
        
        logger.set_log_level_from_string("ERROR")
        self.assertEqual(logger.logger.level, logging.ERROR)
        
        logger.set_log_level_from_string("info")
        self.assertEqual(logger.logger.level, logging.INFO)
        
        # Invalid level should raise ValueError
        with self.assertRaises(ValueError):
            logger.set_log_level_from_string("INVALID")

    def test_redirect_to_stderr_moves_stdout_handler_and_restores_it(self):
        """Test that redirect_to_stderr retargets a stdout StreamHandler to stderr, then back."""
        os.environ["SCALEOUT_LOG_CONSOLE"] = "true"
        logger = ScaleoutLogger()

        handler = next(h for h in logger.logger.handlers if isinstance(h, logging.StreamHandler))
        self.assertIs(handler.stream, sys.stdout)

        with logger.redirect_to_stderr():
            self.assertIs(handler.stream, sys.stderr)

        self.assertIs(handler.stream, sys.stdout)

    def test_redirect_to_stderr_restores_stream_on_exception(self):
        """Test that redirect_to_stderr restores stdout even if the with-block raises."""
        os.environ["SCALEOUT_LOG_CONSOLE"] = "true"
        logger = ScaleoutLogger()
        handler = next(h for h in logger.logger.handlers if isinstance(h, logging.StreamHandler))

        try:
            with logger.redirect_to_stderr():
                self.assertIs(handler.stream, sys.stderr)
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            self.fail("expected ValueError to propagate")

        self.assertIs(handler.stream, sys.stdout)

    def test_redirect_to_stderr_ignores_non_stdout_handlers(self):
        """Test that a handler not writing to stdout (e.g. a stderr handler) is left alone."""
        logger = ScaleoutLogger()
        stderr_handler = logging.StreamHandler(sys.stderr)
        logger.logger.addHandler(stderr_handler)

        with logger.redirect_to_stderr():
            self.assertIs(stderr_handler.stream, sys.stderr)

        self.assertIs(stderr_handler.stream, sys.stderr)

    def test_set_log_stream_to_file(self):
        """Test redirecting logs to a file."""
        import tempfile
        
        logger = ScaleoutLogger()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            logger.set_log_stream(log_file)
            
            # Should have FileHandler
            handler_types = [type(h).__name__ for h in logger.logger.handlers]
            self.assertIn('FileHandler', handler_types)
            
            # Write a log message
            logger.warning("Test message")
            
            # Flush handlers
            for handler in logger.logger.handlers:
                handler.flush()
            
            # Check file content
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)
        finally:
            # Cleanup
            if os.path.exists(log_file):
                os.remove(log_file)


if __name__ == '__main__':
    unittest.main()
