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
import shlex
import subprocess
import sys

from scaleoututil.logging import ScaleoutLogger

_IS_UNIX = os.name != "nt"


class ShellCommandException(Exception):
    @classmethod
    def from_completed_process(cls, process):
        lines = [
            f"Non-zero exit code: {process.returncode}",
            f"Command: {process.args}",
        ]
        if process.stdout:
            lines += [
                "",
                "STDOUT:",
                process.stdout,
            ]
        if process.stderr:
            lines += [
                "",
                "STDERR:",
                process.stderr,
            ]
        return cls("\n".join(lines))


def _join_commands(*commands):
    entry_point = ["bash", "-c"] if _IS_UNIX else ["cmd", "/c"]
    sep = " && " if _IS_UNIX else " & "

    def _to_str(cmd):
        if isinstance(cmd, (list, tuple)):
            parts = [str(p) for p in cmd]
            return shlex.join(parts) if _IS_UNIX else subprocess.list2cmdline(parts)
        return str(cmd)

    return [*entry_point, sep.join(_to_str(c) for c in commands)]


def _exec_cmd(
    cmd,
    *,
    cwd=None,
    throw_on_error=True,
    extra_env=None,
    capture_output=True,
    synchronous=True,
    stream_output=False,
    **kwargs,
):
    """A convenience wrapper of `subprocess.Popen` for running a command from a Python script.

    Args:
    ----
        cmd: The command to run, as a string or a list of strings.
        cwd: The current working directory.
        throw_on_error: If True, raises an Exception if the exit code of the program is nonzero.
        extra_env: Extra environment variables to be defined when running the child process.
            If this argument is specified, `kwargs` cannot contain `env`.
        capture_output: If True, stdout and stderr will be captured and included in an exception
            message on failure; if False, these streams won't be captured.
        synchronous: If True, wait for the command to complete and return a CompletedProcess
            instance, If False, does not wait for the command to complete and return
            a Popen instance, and ignore the `throw_on_error` argument.
        stream_output: If True, stream the command's stdout and stderr to `sys.stdout`
            as a unified stream during execution.
            If False, do not stream the command's stdout and stderr to `sys.stdout`.
        kwargs: Keyword arguments (except `text`) passed to `subprocess.Popen`.

    Returns:
    -------
        If synchronous is True, return a `subprocess.CompletedProcess` instance,
        otherwise return a Popen instance.

    """
    illegal_kwargs = set(kwargs.keys()).intersection({"text"})
    if illegal_kwargs:
        raise ValueError(f"`kwargs` cannot contain {list(illegal_kwargs)}")

    env = kwargs.pop("env", None)
    if extra_env is not None and env is not None:
        raise ValueError("`extra_env` and `env` cannot be used at the same time")

    if capture_output and stream_output:
        raise ValueError("`capture_output=True` and `stream_output=True` cannot be specified at the same time")

    env = env if extra_env is None else {**os.environ, **extra_env}

    if isinstance(cmd, list):
        cmd = list(map(str, cmd))

    if capture_output or stream_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output or stream_output")
        kwargs["stdout"] = subprocess.PIPE
        if capture_output:
            kwargs["stderr"] = subprocess.PIPE
        elif stream_output:
            # Redirect stderr to stdout in order to combine the streams for unified printing to
            # `sys.stdout`, as documented in
            # https://docs.python.org/3/library/subprocess.html#subprocess.run
            kwargs["stderr"] = subprocess.STDOUT

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        **kwargs,
    )
    if not synchronous:
        return process

    if stream_output:
        for output_char in iter(lambda: process.stdout.read(1), ""):
            sys.stdout.write(output_char)

    stdout, stderr = process.communicate()
    returncode = process.poll()
    comp_process = subprocess.CompletedProcess(
        process.args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if throw_on_error and returncode != 0:
        raise ShellCommandException.from_completed_process(comp_process)
    return comp_process


def run_process(args, cwd):
    """Run a process and log the output.

    :param args: The arguments to the process.
    :type args: list
    :param cwd: The current working directory.
    :type cwd: str
    :return:
    """
    status = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def check_io():
        """Check stdout/stderr of the child process.

        :return:
        """
        while True:
            output = status.stdout.readline().decode()
            if output:
                ScaleoutLogger().info(output)
            else:
                break

    # keep checking stdout/stderr until the child exits
    while status.poll() is None:
        check_io()
