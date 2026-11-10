"""Portions of this code are derived from the Apache 2.0 licensed project mlflow (https://mlflow.org/).,
with modifications made by Scaleout Systems AB.
Copyright (c) 2018 Databricks, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout as FileLockTimeout

from scaleoututil.config import SCALEOUT_CLIENT_INSTALL_SPEC, SCALEOUT_UTIL_INSTALL_SPEC
import yaml

from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils import PYTHON_VERSION
from scaleoututil.utils.dist import get_version
from scaleoututil.utils.process import _exec_cmd, _join_commands

_REQUIREMENTS_FILE_NAME = "requirements.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"
_PYTHON_ENV_METADATA_FILE_NAME = "env_metadata.txt"
_IS_UNIX = os.name != "nt"


def _pip_install_cmd(requirements_file: str) -> list:
    """Return a pip install command that works in uv-managed venvs (no pip by default).

    Prefers ``uv pip install`` when pip is absent from the current interpreter,
    falling back to ``python -m pip install``.
    """
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    uv_exe = shutil.which("uv")
    if uv_exe:
        return [uv_exe, "pip", "install", "--python", sys.executable, "-r", requirements_file]
    raise RuntimeError(
        "pip is not available in the current environment and 'uv' was not found on PATH. "
        "Install pip (pip install pip) or install uv (https://github.com/astral-sh/uv)."
    )


def get_executable_path_from_venv_path(path: Path):
    """Get the path to the Python interpreter from an environment."""
    paths = ("bin", "python") if _IS_UNIX else ("Scripts", "python.exe")
    return Path(path).joinpath(*paths)


class PythonEnv:
    def __init__(self, name=None, python=None, scaleout=None, build_dependencies=None, dependencies=None):
        """Represents environment information for Scaleout compute packages.

        Args:
        ----
            name: Name of environment. If unspecified, defaults to fedn_env
            python: Python version for the environment. If unspecified, defaults to the current
                Python version.
            scaleout: Scaleout version to be installed in the environment, default to the current scaleout version
            build_dependencies: List of build dependencies for the environment that must
                be installed before installing ``dependencies``. If unspecified,
                defaults to an empty list.
            dependencies: List of dependencies for the environment. If unspecified, defaults to
                an empty list.

        """
        if name is not None and not isinstance(name, str):
            raise TypeError(f"`name` must be a string but got {type(name)}")
        if python is not None and not isinstance(python, str):
            raise TypeError(f"`python` must be a string but got {type(python)}")
        if build_dependencies is not None and not isinstance(build_dependencies, list):
            raise TypeError(f"`build_dependencies` must be a list but got {type(build_dependencies)}")
        if dependencies is not None and not isinstance(dependencies, list):
            raise TypeError(f"`dependencies` must be a list but got {type(dependencies)}")
        self._name = name
        self.python = python or PYTHON_VERSION
        self.scaleout_version = scaleout or get_version("scaleout")
        self.build_dependencies = build_dependencies or []
        self.dependencies = dependencies or []
        self._base_path = None
        self._path = None

        self.remove_scaleoutdependency()

    def set_path(self, path):
        """Set the full path to the environment."""
        self._path = path

    def set_base_path(self, path):
        self._base_path = path

    @property
    def path(self) -> Path:
        """Get the full path to the environment."""
        if self._path:
            return Path(self._path)
        if not self._base_path:
            raise ValueError("Base path is not set. Use `set_base_path` to set it.")
        return Path(self._base_path).joinpath(self.name)

    @property
    def name(self) -> str:
        if self._name is None:
            return f".venv-{self.get_sha()}"
        else:
            return f"{self._name}-{self.get_sha()}"

    def __str__(self):
        return str(self.to_dict())

    def get_sha(self):
        """Returns a SHA256 hash of the environment configuration."""
        imporant_features = {"python": self.python, "scaleout": self.scaleout_version, "build_deps": self.build_dependencies, "deps": self.dependencies}
        env_str = str(imporant_features).encode("utf-8")
        return hashlib.sha256(env_str).hexdigest()

    def remove_scaleoutdependency(self):
        """Remove 'fedn' and 'scaleout' from dependencies if it exists."""
        self.dependencies = [dep for dep in self.dependencies if dep != "fedn" and dep != "scaleout" and dep != "scaleoututil"]
        self.build_dependencies = [dep for dep in self.build_dependencies if dep != "fedn" and dep != "scaleout" and dep != "scaleoututil"]

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_yaml(self, path):
        with open(path, "w") as f:
            # Exclude None and empty lists
            data = {k: v for k, v in self.to_dict().items() if v}
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @staticmethod
    def get_dependencies_from_conda_yaml(path):
        raise NotImplementedError

    @classmethod
    def from_conda_yaml(cls, path):
        return cls.from_dict(cls.get_dependencies_from_conda_yaml(path))

    def get_activate_cmd(self):
        """Get the command to activate the environment."""
        paths = ("bin", "activate") if _IS_UNIX else ("Scripts", "activate.bat")
        activate_cmd = self.path.joinpath(*paths)
        activate_cmd = f"source {activate_cmd}" if _IS_UNIX else str(activate_cmd)
        return activate_cmd

    def get_executable(self):
        return get_executable_path_from_venv_path(self.path)

    def verify_installed_env(self, path: Path = None):
        """Check if the environment metadata file exists and matches the environment installed at this path"""
        if path is None:
            path = self.path
        return self._verify_metadata(path)

    def _write_metadata(self, env_dir: Path):
        build_deps = "\n".join(self.build_dependencies or [])
        deps = "\n".join(self.dependencies or [])
        Path(env_dir).joinpath(_PYTHON_ENV_METADATA_FILE_NAME).write_text(f"{self.get_sha()}\n{self.python}\n{self.scaleout_version}\n{build_deps}\n{deps}")

    def _verify_metadata(self, env_dir: Path):
        metadata_file = env_dir.joinpath(_PYTHON_ENV_METADATA_FILE_NAME)
        if not metadata_file.exists():
            return False

        with open(metadata_file) as f:
            sha, python_version, scaleout_version, *deps = f.read().splitlines()
            if sha != self.get_sha() or python_version != self.python or scaleout_version != self.scaleout_version:
                return False

            # Check if dependencies match
            if set(deps) != set(self.build_dependencies + self.dependencies):
                return False
        return True

    def install_into_current_env(self, capture_output=False):
        """Install the dependencies into the current environment."""
        extra_env = {
            # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
            # otherwise pip might prompt "yes or no" and ask stdin input
            "PIP_NO_INPUT": "1",
        }

        ScaleoutLogger().info("Installing dependencies into the current environment")
        for deps in filter(None, [self.build_dependencies, self.dependencies]):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(deps))
                cmd = _pip_install_cmd(tmp_req_file)
                _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)
        ScaleoutLogger().info("Writeing metadata to " + self.path.as_posix())
        self._write_metadata(self.path)

        return True

    def create_virtualenv(self, capture_output=False):
        env_dir = self.path

        if env_dir.exists():
            ScaleoutLogger().info("Environment %s already exists", env_dir)
            return True

        # Creating the environment is not atomic: the existence check above and
        # the virtualenv + pip work below are separated by minutes. Processes
        # sharing a project directory (e.g. the N siblings spawned by
        # scaleoututil.launchers.launch_clients) all resolve to the same env_dir,
        # so without a lock they build it concurrently -- and the loser's
        # remove_on_error cleanup rmtree's the winner's environment mid-install.
        # The lock file lives next to env_dir so it survives that cleanup.
        os.makedirs(env_dir.parent, exist_ok=True)
        lock = FileLock(str(env_dir.parent / f"{env_dir.name}.lock"))
        try:
            lock.acquire(timeout=0)
        except FileLockTimeout:
            ScaleoutLogger().info("Another process is creating the environment in %s, waiting for it to finish", env_dir)
            lock.acquire()

        try:
            # Re-check under the lock: another process may have finished
            # building the environment while we waited for it.
            if env_dir.exists():
                ScaleoutLogger().info("Environment %s already exists", env_dir)
                return True
            return self._create_virtualenv_locked(env_dir, capture_output=capture_output)
        finally:
            lock.release()

    def _create_virtualenv_locked(self, env_dir: Path, capture_output=False):
        """Build the environment at ``env_dir``; the caller must hold the creation lock.

        ``remove_on_error`` below deletes ``env_dir`` on failure. That is only
        safe because the caller established, under the lock, that this process
        is the one creating it -- otherwise the cleanup would delete an
        environment another process is installing into or already using.
        """
        activate_cmd = self.get_activate_cmd()

        with remove_on_error(
            env_dir,
            onerror=lambda e: ScaleoutLogger().warning(
                "Encountered an unexpected error: %s while creating a virtualenv environment in %s, removing the environment directory...",
                repr(e),
                env_dir,
            ),
        ):
            os.makedirs(env_dir, exist_ok=True)
            ScaleoutLogger().info("Creating a new environment in %s with %s", env_dir, sys.executable)
            _exec_cmd(
                [sys.executable, "-m", "virtualenv", "--python", sys.executable] + [env_dir],
                capture_output=capture_output,
            )

            extra_env = {
                # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
                # otherwise pip might prompt "yes or no" and ask stdin input
                "PIP_NO_INPUT": "1",
            }

            # Install build deps
            ScaleoutLogger().info("Installing build dependecies")
            if self.build_dependencies:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                    Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(self.build_dependencies))
                    cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                    _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

            # Install scaleout. SCALEOUT_UTIL_INSTALL_SPEC / SCALEOUT_CLIENT_INSTALL_SPEC let
            # callers override the pip spec (e.g. a local path or `-e /path`) for dev installs.
            # Util and client are version-locked, so they must be overridden together.
            util_spec = SCALEOUT_UTIL_INSTALL_SPEC
            client_spec = SCALEOUT_CLIENT_INSTALL_SPEC
            if bool(util_spec) != bool(client_spec):
                raise ValueError("SCALEOUT_UTIL_INSTALL_SPEC and SCALEOUT_CLIENT_INSTALL_SPEC must be set together")

            if util_spec and client_spec:
                ScaleoutLogger().info("Installing scaleoututil from %s and scaleout from %s", util_spec, client_spec)
                cmd = _join_commands(activate_cmd, f"python -m pip install {util_spec} {client_spec}")
                _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)
            else:
                scaleout_version = self.scaleout_version
                if scaleout_version == "latest":
                    ScaleoutLogger().info("Installing scaleout")
                    cmd = _join_commands(activate_cmd, "python -m pip install scaleout")
                    _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)
                elif scaleout_version != "unknown":
                    ScaleoutLogger().info("Installing scaleout==%s", scaleout_version)
                    cmd = _join_commands(activate_cmd, f"python -m pip install scaleout=={scaleout_version}")
                    _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)
                else:
                    ScaleoutLogger().warning("Could not determine scaleout version; installing latest")
                    cmd = _join_commands(activate_cmd, "python -m pip install scaleout")
                    _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)

            ScaleoutLogger().info("Installing package dependencies")
            if self.dependencies:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                    Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(self.dependencies))
                    cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                    _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

            self._write_metadata(env_dir)

        return True

    def validate_scaleout_install(self) -> bool:
        """Verify that the environment can import scaleout and run `scaleout --version`.

        Returns True if both checks succeed, False otherwise.
        """
        python_executable = str(get_executable_path_from_venv_path(self.path))

        try:
            _exec_cmd([python_executable, "-c", "import scaleout"], capture_output=True)
        except Exception as e:
            ScaleoutLogger().error(f"Environment at {self.path} cannot import scaleout: {e}")
            return False

        try:
            result = _exec_cmd([python_executable, "-m", "scaleout", "--version"], capture_output=True)
        except Exception as e:
            ScaleoutLogger().error(f"`scaleout --version` failed in environment at {self.path}: {e}")
            return False

        ScaleoutLogger().info(f"Environment at {self.path} is valid: {result.stdout.strip()}")
        return True


@contextmanager
def remove_on_error(path: os.PathLike, onerror=None):
    """A context manager that removes a file or directory if an exception is raised during
    execution.
    """
    try:
        yield
    except Exception as e:
        if onerror:
            onerror(e)
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        raise
