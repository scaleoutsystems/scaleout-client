import contextlib
import logging
import os
import sys


class ScaleoutLogger:
    """A simple logger wrapper for Scaleout."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ScaleoutLogger, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.logger = logging.getLogger("scaleout")

        # Check if logging should be enabled (for CLI tools or explicit opt-in)
        # Default to WARNING for library usage, can be overridden by environment variable
        log_level_str = os.environ.get("SCALEOUT_LOG_LEVEL", "WARNING").upper()
        enable_console = os.environ.get("SCALEOUT_LOG_CONSOLE", "false").lower() == "true"

        # If no handlers exist, add appropriate handler
        if not self.logger.handlers:
            if enable_console or log_level_str != "WARNING":
                # Add console handler if explicitly requested or if custom log level set
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(self._formatter)
                self.logger.addHandler(handler)
            else:
                # Use NullHandler for library usage (best practice)
                self.logger.addHandler(logging.NullHandler())

        # Set log level from environment or default
        try:
            self.set_log_level_from_string(log_level_str)
        except ValueError:
            self.logger.setLevel(logging.WARNING)
            self.logger.warning(f"Invalid log level '{log_level_str}', defaulting to WARNING")

        self.logger.propagate = False
        self.__initialized = True

    @property
    def _formatter(self):
        return logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.logger.critical(*args, **kwargs)

    def set_log_level_from_string(self, level_str):
        """Set the log level based on a string input."""
        level_str = level_str.upper()
        # mapping = logging.getLevelNamesMapping() this is available in Python 3.11+
        mapping = logging._nameToLevel  # for compatibility with older versions
        if level_str not in mapping:
            raise ValueError(f"Invalid log level: {level_str}")
        self.logger.setLevel(mapping[level_str])

    def set_log_stream(self, log_file=None):
        """Redirect logs to a file if log_file is provided, otherwise to stdout."""
        # Remove existing Stream, File, and Null handlers
        for handler in list(self.logger.handlers):
            if isinstance(handler, (logging.StreamHandler, logging.FileHandler, logging.NullHandler)):
                self.logger.removeHandler(handler)
        if log_file:
            # Add a FileHandler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._formatter)
            self.logger.addHandler(file_handler)
        else:
            # Add a StreamHandler for stdout
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self._formatter)
            self.logger.addHandler(stream_handler)

    def enable_console_logging(self, level=logging.INFO):
        """Enable console logging at the specified level.

        Useful for CLI applications or when you want to see logs in library usage.

        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)

        """
        # Remove NullHandler if present
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.NullHandler):
                self.logger.removeHandler(handler)

        # Add StreamHandler if not already present
        has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        if not has_stream_handler:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(self._formatter)
            self.logger.addHandler(stream_handler)

        self.logger.setLevel(level)

    @contextlib.contextmanager
    def redirect_to_stderr(self):
        """Temporarily redirect any handler currently writing to stdout to stderr.

        For CLI commands whose contract is a single clean value on stdout, safe for
        `$(...)` capture (e.g. `scaleout client create-enrollment-token`, `scaleout
        client enroll`) — so log messages a caller opted into via SCALEOUT_LOG_LEVEL/
        SCALEOUT_LOG_CONSOLE land on stderr instead of corrupting that captured value,
        rather than being silently dropped.
        """
        handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout]
        for h in handlers:
            h.setStream(sys.stderr)
        try:
            yield
        finally:
            for h in handlers:
                h.setStream(sys.stdout)

    def add_handler(self, handler):
        """Add a custom handler to the logger."""
        self.logger.addHandler(handler)

    def shutdown_logging(self):
        """Shutdown the logger to ensure logs are flushed before exit."""
        logging.shutdown()
