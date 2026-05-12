"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

import os
import sys
from pathlib import Path
import traceback
from typing import Optional

from scaleoututil.config import SCALEOUT_ARCHIVE_DIR, SCALEOUT_PACKAGE_EXTRACT_DIR
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.package_runtime import PackageRuntime
from scaleoututil.utils.environment import PythonEnv

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    full_package_path = os.path.join(os.getcwd(), SCALEOUT_PACKAGE_EXTRACT_DIR)
    full_archive_path = os.path.join(os.getcwd(), SCALEOUT_ARCHIVE_DIR)

    os.makedirs(full_package_path, exist_ok=True)
    os.makedirs(full_archive_path, exist_ok=True)

    return full_package_path, full_archive_path


class ImporterPackageRuntime(PackageRuntime):
    """ImporterPackageRuntime is used to download, validate, and unpack compute packages.

    :param package_path: Path to compute package.
    :type package_path: str
    """

    def __init__(self, package_path: str, archive_path: str) -> None:
        """Initialize the PackageRuntime."""
        super().__init__(package_path, archive_path)
        self.python_env: Optional[PythonEnv] = None
        self.requires_restart = False

    @property
    def active_env_path(self):
        return sys.prefix

    def _init_runtime(self):
        """Initialize the Python environment."""
        if self.config is None:
            ScaleoutLogger().error("Package runtime is not loaded.")
            return False
        try:
            python_env_yaml_path = self.config.get("python_env")
            if python_env_yaml_path:
                python_env_yaml_path = Path(self._target_path).joinpath(python_env_yaml_path)
                ScaleoutLogger().info(f"Reading environment configuration from: {python_env_yaml_path}")
                self.python_env = PythonEnv.from_yaml(python_env_yaml_path)
                self.python_env.remove_scaleoutdependency()
                self.python_env.set_path(self.active_env_path)
            else:
                ScaleoutLogger().info("No environment configuration specified in config")
                self.python_env = None
        except Exception as e:
            ScaleoutLogger().error(f"Error initializing environment configuration: {e}")
            self.python_env = None
            return False
        return True

    def update_runtime_env(self):
        if "python_env" not in self.config:
            ScaleoutLogger().error("Python environment is not specified in the package configuration. Package runtime cannot be managed.")
            raise RuntimeError("Python environment is not specified in the package configuration. Package runtime cannot be managed.")
        else:
            if not self._init_runtime():
                ScaleoutLogger().error("Failed to initialize the managed environment.")
                raise RuntimeError("Failed to initialize the managed environment.")
            if self.python_env is None:
                ScaleoutLogger().info("No managed environment specified, running in current environment. Managed environment flag will be ignored.")
            else:
                if not self._verify_active_environment():
                    ScaleoutLogger().error("Active environment cannot be managed: {}".format(self.active_env_path))
                    raise RuntimeError("Active environment cannot be managed.")

                if not self._check_and_install_runtime_environment():
                    ScaleoutLogger().error("Failed to verify or install the managed environment.")
                    raise RuntimeError("Failed to verify or install the managed environment.")

    def _verify_active_environment(self) -> bool:
        """Verify the Python environment."""
        if self.active_env_path is None or not self.active_env_path:
            ScaleoutLogger().error("A managed environment requires a virtual environment to be active.")
            return False
        env_path = self.active_env_path
        ScaleoutLogger().info(f"Virtual environment detected at: {env_path}")
        if Path(env_path) != Path(self.python_env.path):
            ScaleoutLogger().warning(f"Virtual environment path {env_path} does not match the expected path {self.python_env.path}.")
            return False
        return True

    def _check_and_install_runtime_environment(self) -> bool:
        """Verify that the environment is set up correctly."""
        if self.config is None:
            ScaleoutLogger().error("Package runtime is not loaded.")
            return False
        if not self.active_env_path:
            ScaleoutLogger().error("No virtual environment specified, cannot initialize environment.")
            return False
        if self.python_env is None:
            ScaleoutLogger().error("No environment specified in config")
            return False
        try:
            if self.python_env.verify_installed_env():
                ScaleoutLogger().info("Current environment is up to date")
                return True
            else:
                self.requires_restart = self._install_runtime_environment()
                return True
        except Exception as e:
            ScaleoutLogger().error(f"Error in checking or updating python environment: {e}")
            self.python_env = None
            return False

    def _install_runtime_environment(self) -> bool:
        """Install the environment if needed.

        Returns True if the environment was updated, False otherwise.
        """
        if self.python_env.verify_installed_env():
            ScaleoutLogger().info("Python environment is already up to date")
            return False
        else:
            if not self.active_env_path:
                ScaleoutLogger().error("No virtual environment specified, cannot update the environment.")
                raise RuntimeError("No virtual environment specified, cannot update the environment.")
            if not os.path.abspath(self.active_env_path).startswith(os.path.abspath(os.getcwd())):
                ScaleoutLogger().error("The specified virtual environment must reside inside the current working directory.")
                raise RuntimeError("The specified virtual environment must reside inside the current working directory.")
            self.python_env.install_into_current_env(capture_output=True)
            if not self.python_env.verify_installed_env():
                ScaleoutLogger().error(f"Python environment at {self.python_env.path} could not be verified after installation.")
                raise RuntimeError("Failed to update the environment.")
            return True

    def run_entrypoint(self, entrypoint: str, *args, **kwargs) -> bool:
        """Run a specified entrypoint from the package configuration."""
        if self.config is None:
            ScaleoutLogger().error("Package runtime is not initialized.")
            return False

        original_sys_path = sys.path.copy()
        try:
            # Add the package path to sys.path
            sys.path.insert(0, self._target_path)
            entrypoints = self.config.get("entry_points")
            if entrypoints:
                entrypoint_py = entrypoints.get(entrypoint)
            else:
                entrypoint_py = None
            if not entrypoint_py:
                ScaleoutLogger().error(f"No '{entrypoint}' entrypoint defined in the configuration.")
                return False

            if not Path(self._target_path).joinpath(entrypoint_py).exists():
                ScaleoutLogger().error(f"Entrypoint script {entrypoint_py} not found in the package directory.")
                raise FileNotFoundError(f"Entrypoint script {entrypoint_py} not found.")

            entrypoint_module = Path(self._target_path).joinpath(entrypoint_py).stem
            ScaleoutLogger().info(f"Running entrypoint '{entrypoint}' from: {entrypoint_module}")
            try:
                module = __import__(entrypoint_module)
                if hasattr(module, entrypoint):
                    func = getattr(module, entrypoint)
                    func(*args, **kwargs)
                else:
                    ScaleoutLogger().error(f"Entrypoint function '{entrypoint}' not found in module '{entrypoint_module}'.")
                    return False
            except Exception as e:
                ScaleoutLogger().error(f"Error executing entrypoint '{entrypoint}': {e}")
                traceback.print_exc()
                return False
        except Exception as e:
            ScaleoutLogger().error(f"Error during running entrypoint '{entrypoint}': {e}")
            return False
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path

        return True

    def run_startup(self, edge_client):
        """Run the client startup script."""
        return self.run_entrypoint("startup", edge_client)
