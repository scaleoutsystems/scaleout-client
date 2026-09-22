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

import os
from pathlib import Path
import shutil
from contextlib import contextmanager

from scaleoututil.config import SCALEOUT_VENV_DIR
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.environment import PythonEnv
from scaleoututil.utils.process import _exec_cmd, _join_commands

_IS_UNIX = os.name != "nt"


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


def _install_python(version, pyenv_root=None, capture_output=False):
    """Installs a specified version of python with pyenv and returns a path to the installed python
    binary.
    """
    raise NotImplementedError("This function is not implemented yet.")


def _is_virtualenv_available():
    """Returns True if virtualenv is available, otherwise False."""
    return shutil.which("virtualenv") is not None


def _get_python_env(python_env_file) -> PythonEnv:
    """Parses a python environment file and returns a dictionary with the parsed content."""
    if os.path.exists(python_env_file):
        return PythonEnv.from_yaml(python_env_file)


class Dispatcher:
    """Dispatcher class for compute packages.

    :param config: The configuration.
    :type config: dict
    :param dir: The directory to dispatch to.
    :type dir: str
    """

    def __init__(self, config, project_dir):
        """Initialize the dispatcher."""
        self.config = config
        self.project_dir = project_dir
        self.activate_cmd = ""
        self.python_env_path = ""

    def get_or_create_python_env(self, capture_output=False):
        python_env = self.config.get("python_env", "")
        if not python_env:
            ScaleoutLogger().info("No python_env specified in the configuration, using the system Python.")
            self.activate_cmd = ""
            return self.activate_cmd
        else:
            python_env_yaml_path = os.path.join(self.project_dir, python_env)
            if not os.path.exists(python_env_yaml_path):
                raise Exception("Compute package specified python_env file %s, but no such file was found." % python_env_yaml_path)
            python_env = _get_python_env(python_env_yaml_path)

        python_env.set_base_path(Path(self.project_dir) / SCALEOUT_VENV_DIR)
        if not python_env.path.exists():
            python_env.create_virtualenv(capture_output=capture_output)
        elif python_env.verify_installed_env():
            ScaleoutLogger().info("Using existing virtualenv at %s", python_env.path)
        else:
            ScaleoutLogger().error("Found environment but it is not active. Please clear cache.")
            raise RuntimeError("Found environment but it is not active. Please clear cache.")

        self.activate_cmd = python_env.get_activate_cmd()
        self.python_env_path = python_env.path
        return self.activate_cmd

    def run_cmd(self, cmd_type, capture_output=False, extra_env=None, synchronous=True, stream_output=False):
        """Run a command.

        :param cmd_type: The command type.
        :type cmd_type: str
        :return:
        """
        try:
            cmdsandargs = cmd_type.split(" ")

            entry_point = self.config["entry_points"][cmdsandargs[0]]["command"]

            # remove the first element,  that is not a file but a command
            args = cmdsandargs[1:]

            # Join entry point and arguments into a single command as a string
            entry_point_args = " ".join(args)
            entry_point = f"{entry_point} {entry_point_args}"

            if self.activate_cmd:
                cmd = _join_commands(self.activate_cmd, entry_point)
            else:
                cmd = _join_commands(entry_point)

            ScaleoutLogger().info("Running command: {}".format(cmd))
            _exec_cmd(
                cmd,
                cwd=self.project_dir,
                throw_on_error=True,
                extra_env=extra_env,
                capture_output=capture_output,
                synchronous=synchronous,
                stream_output=stream_output,
            )

            ScaleoutLogger().info("Done executing {}".format(cmd_type))
        except IndexError:
            message = "No such argument or configuration to run."
            ScaleoutLogger().error(message)

    def delete_virtual_environment(self):
        if self.python_env_path:
            ScaleoutLogger().info(f"Removing virtualenv {self.python_env_path}")
            shutil.rmtree(self.python_env_path)
        else:
            ScaleoutLogger().warning("No virtualenv found to remove.")
