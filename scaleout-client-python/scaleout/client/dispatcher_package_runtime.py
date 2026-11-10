"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

from typing import Optional

from scaleoututil.logging import ScaleoutLogger
from scaleout.client.package_runtime import PackageRuntime
from scaleoututil.utils.dispatcher import Dispatcher
import traceback

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


class DispatcherPackageRuntime(PackageRuntime):
    """PackageRuntime is used to download, validate, and unpack compute packages.

    :param package_path: Path to compute package.
    :type package_path: str
    """

    def __init__(self, package_path: str, archive_path: str) -> None:
        """Initialize the PackageRuntime."""
        super().__init__(package_path, archive_path)

        self.dispatcher: Optional[Dispatcher] = None
        self._initialized = False

    def init_runtime(self):
        if self.config is None:
            ScaleoutLogger().error("Package is not loaded")
            return False

        try:
            self.dispatcher = Dispatcher(self.config, self._target_path)
        except Exception as e:
            ScaleoutLogger().error(f"Error setting dispatcher: {e}")
            return False
        self._initialized = True
        return True

    def run_startup(self):
        if not self._initialized:
            if not self.init_runtime():
                return False

        try:
            ScaleoutLogger().info("Initiating Dispatcher with entrypoint set to: startup")
            activate_cmd = self.dispatcher.get_or_create_python_env()
            self.dispatcher.run_cmd("startup")
        except KeyError:
            ScaleoutLogger().info("No startup command found in package. Continuing.")
        except Exception as e:
            traceback.print_exc()
            ScaleoutLogger().error(f"Caught exception: {type(e).__name__}")
            return False

        if activate_cmd:
            ScaleoutLogger().info(f"To activate the virtual environment, run: {activate_cmd}")

        return True
