"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

import os
import sys
from pathlib import Path
import traceback
from typing import Optional

from scaleoututil.config import SCALEOUT_ARCHIVE_DIR, SCALEOUT_PACKAGE_EXTRACT_DIR, SCALEOUT_VENV_DIR
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.package_runtime import PackageRuntime
from scaleoututil.utils.environment import PythonEnv
from scaleoututil.utils.process import _exec_cmd, _join_commands

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
        self._initialized = False

    @property
    def active_env_path(self):
        return sys.prefix

    @property
    def has_configuration(self):
        if self._initialized:
            return self.python_env is not None
        return False

    def init_runtime(self):
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
            else:
                ScaleoutLogger().info("No environment configuration specified in config")
                self.python_env = None
        except Exception as e:
            ScaleoutLogger().error(f"Error initializing environment configuration: {e}")
            self.python_env = None
            return False
        self._initialized = True
        return True

    def update_current_runtime_env(self):
        if not self._initialized:
            if not self.init_runtime():
                ScaleoutLogger().error("Failed to initialize environment configuration")
                raise RuntimeError("Failed to initialize environment configuration")
        if self.python_env is None:
            ScaleoutLogger().info("No environment configuration")
        else:
            # Set current venv as target
            ScaleoutLogger().info("Using venv at: " + self.active_env_path)
            self.python_env.set_path(self.active_env_path)

            if not self._check_and_install_runtime_environment():
                ScaleoutLogger().error("Failed to verify or install the managed environment.")
                raise RuntimeError("Failed to verify or install the managed environment.")

    def create_runtime_env(self) -> bool:
        if not self._initialized:
            if not self.init_runtime():
                ScaleoutLogger().error("Failed to initialize environment configuration")
                raise RuntimeError("Failed to initialize environment configuration")
        if self.python_env is None:
            ScaleoutLogger().info("No environment configuration")
            return False
        else:
            self.python_env.set_base_path(Path(os.getcwd()) / SCALEOUT_VENV_DIR)
            if self.python_env.verify_installed_env():
                if Path(self.active_env_path) == self.python_env.path:
                    return True
                else:
                    ScaleoutLogger().info(
                        f"Current interpreter {self.active_env_path} and requested interpreter {self.python_env.path} does not match, requires restart"
                    )
                    self.requires_restart = True
                    return True
            self.python_env.create_virtualenv()
            if not self.python_env.verify_installed_env():
                ScaleoutLogger().error("Could not verify the installed environment")
                raise RuntimeError("Could not verify the installed environment")
            if not self.python_env.validate_scaleout_install():
                ScaleoutLogger().error("Failed to run scaleout in the created environment")
                raise RuntimeError("Failed to run scaleout in the created environment")
            self.requires_restart = True
            return True

    def is_current_venv_valid(self):
        ScaleoutLogger().info(Path(self.active_env_path))

        return self.python_env.verify_installed_env(Path(self.active_env_path))

    def _check_and_install_runtime_environment(self) -> bool:
        """Verify that the environment is set up correctly."""
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

    def dispatch_entrypoint(self, entrypoint: str, extra_env=None, capture_output=False, stream_output=False) -> bool:
        """Dispatch a specified entrypoint in a new process inside the managed environment."""
        if self.config is None:
            ScaleoutLogger().error("Package runtime is not initialized.")
            return False

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
            return False

        if self.python_env is None:
            ScaleoutLogger().error("No managed environment is configured; cannot dispatch entrypoint.")
            return False

        entrypoint_module = Path(entrypoint_py).stem
        ScaleoutLogger().info(f"Dispatching entrypoint '{entrypoint}' from module: {entrypoint_module}")

        python_code = f"import sys; sys.path.insert(0, {self._target_path!r}); import {entrypoint_module}; getattr({entrypoint_module}, {entrypoint!r})()"
        command = ["python", "-c", python_code]

        try:
            self._dispatch_command(
                command,
                extra_env=extra_env,
                capture_output=capture_output,
                stream_output=stream_output,
            )
        except Exception as e:
            ScaleoutLogger().error(f"Error dispatching entrypoint '{entrypoint}': {e}")
            return False
        return True

    def _dispatch_command(self, command: list, capture_output=False, extra_env=None, synchronous=True, stream_output=False):
        """Run a command.

        :param cmd_type: The command type.
        :type cmd_type: str
        :return:
        """
        try:
            # Join entry point and arguments into a single command as a string
            cmd = _join_commands(self.python_env.get_activate_cmd(), command)

            ScaleoutLogger().info("Running command: {}".format(cmd))
            _exec_cmd(
                cmd,
                throw_on_error=True,
                extra_env=extra_env,
                capture_output=capture_output,
                synchronous=synchronous,
                stream_output=stream_output,
            )

            ScaleoutLogger().info("Done executing command")
        except Exception:
            ScaleoutLogger().error("Command exection failed")
            raise

    def run_startup(self, edge_client):
        """Run the client startup script."""
        return self.run_entrypoint("startup", edge_client)
