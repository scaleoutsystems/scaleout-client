import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Optional

import requests

from scaleoututil.config import SCALEOUT_ARCHIVE_DIR, SCALEOUT_AUTH_SCHEME, SCALEOUT_CONNECT_API_SECURE, SCALEOUT_PACKAGE_EXTRACT_DIR
from scaleoututil.logging import ScaleoutLogger
from scaleoututil.utils.environment import get_executable_path_from_venv_path
from scaleoututil.utils.http_status_codes import HTTP_STATUS_NO_CONTENT, HTTP_STATUS_OK
from scaleoututil.utils.checksum import sha
from scaleoututil.utils.yaml import read_yaml_file

REQUEST_TIMEOUT = 10  # seconds  # Default timeout for requests


def parse_header(value):
    """Robust RFC 6266 Content-Disposition parser.
    Returns (main_value, params_dict).
    """
    if not value:
        return "", {}

    parts = [p.strip() for p in str(value).split(";") if p.strip()]

    main = parts[0].lower() if parts else ""

    params = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip().strip('"')
            params[k] = v

    return main, params


def _snapshot_files(path: str) -> set[str]:
    """Return relative paths of all files currently under ``path``."""
    if not path or not os.path.isdir(path):
        return set()
    found: set[str] = set()
    for root, _, files in os.walk(path):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), path)
            found.add(os.path.normpath(rel))
    return found


def _remove_files(base: str, rel_paths: set[str]) -> None:
    """Remove the given files (relative to ``base``) and any empty directories left behind."""
    for rel in rel_paths:
        full = os.path.join(base, rel)
        try:
            os.remove(full)
        except OSError as e:
            ScaleoutLogger().warning(f"Could not remove {full}: {e}")
    for root, _, _ in os.walk(base, topdown=False):
        if root == base:
            continue
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError as e:
            ScaleoutLogger().error(f"Failed to clear files: {e}")


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    full_package_path = os.path.join(os.getcwd(), SCALEOUT_PACKAGE_EXTRACT_DIR)
    full_archive_path = os.path.join(os.getcwd(), SCALEOUT_ARCHIVE_DIR)

    os.makedirs(full_package_path, exist_ok=True)
    os.makedirs(full_archive_path, exist_ok=True)

    return full_package_path, full_archive_path


class PackageRuntime:
    def __init__(self, package_path: str = None, archive_path: str = None) -> None:
        """Initialize the PackageRuntime."""
        self.pkg_path = package_path
        self.tar_path = archive_path or os.path.join(os.getcwd(), SCALEOUT_ARCHIVE_DIR)
        os.makedirs(self.tar_path, exist_ok=True)

        self.pkg_name: Optional[str] = None
        self._checksum: Optional[str] = None

        self._target_path: Optional[str] = None
        self._target_name = "scaleout.yaml"

        self.config = None

    @property
    def package_path(self) -> Optional[str]:
        """Get the path to the unpacked compute package."""
        return self._target_path

    def _download_compute_package(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Download compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if download was successful, False otherwise.
        :rtype: bool
        """
        try:
            url = f"{url}/api/v1/packages/download?name={name}" if name else f"{url}/api/v1/packages/download"
            with requests.get(
                url, stream=True, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {token}"}, verify=SCALEOUT_CONNECT_API_SECURE
            ) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    params = parse_header(r.headers.get("Content-Disposition", ""))[-1]
                    try:
                        self.pkg_name = params["filename"]
                        r.raise_for_status()
                        with open(os.path.join(self.tar_path, self.pkg_name), "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return True

                    except KeyError:
                        ScaleoutLogger().error("No package returned.")
                        return False
                else:
                    ScaleoutLogger().error(f"Failed to download package: {r.status_code} {r.reason}")
                    return False
        except Exception as e:
            ScaleoutLogger().error(f"Unknown error downloading package: {e}")
            return False

    def _fetch_package_checksum(self, url: str, token: str) -> bool:
        """Get checksum of compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if checksum was set successfully, False otherwise.
        :rtype: bool
        """
        try:
            path = f"{url}/api/v1/packages/checksum?name={self.pkg_name}"
            with requests.get(
                path, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {token}"}, verify=SCALEOUT_CONNECT_API_SECURE
            ) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    data = r.json()
                    try:
                        self._checksum = data["checksum"]
                    except KeyError:
                        ScaleoutLogger().error("Could not extract checksum.")
            return True
        except Exception:
            return False

    def validate_compute_package(self, url: str, token: str) -> bool:
        """Validate the package against the checksum provided by the controller.

        :param expected_checksum: Checksum provided by the controller.
        :return: True if checksums match, False otherwise.
        :rtype: bool
        """
        try:
            file_checksum = str(sha(os.path.join(self.tar_path, self.pkg_name)))
        except FileNotFoundError:
            ScaleoutLogger().error(f"Package file {self.pkg_name} not found in {self.tar_path}.")
            return False

        success = self._fetch_package_checksum(url, token)
        if not success:
            ScaleoutLogger().error("Failed to fetch package checksum from controller.")
            return False

        if self._checksum == file_checksum:
            ScaleoutLogger().info(f"Package validated {self._checksum}")
            return True
        return False

    def _unpack_compute_package(self, clear_untracked: bool = False) -> Optional[str]:
        """Unpack the compute package.

        :param clear_untracked: If True, remove pre-existing files in the package
            directory that are not part of the newly extracted archive.
        :return: Tuple containing a boolean indicating success and the path to the unpacked package.
        :rtype: Tuple[bool, str]
        """
        if not self.pkg_name:
            ScaleoutLogger().error("Failed to unpack compute package, no pkg_name set. Has the reducer been configured with a compute package?")
            return False, ""

        try:
            if self.pkg_name.endswith(("tar.gz", ".tgz", "tar.bz2")):
                tar_path = os.path.join(self.tar_path, self.pkg_name)
                pre_existing = _snapshot_files(self.pkg_path)

                extracted: set[str] = set()
                with tarfile.open(tar_path, "r:*") as f:
                    for member in f.getmembers():
                        f.extract(member, self.pkg_path)
                        if member.isfile():
                            extracted.add(os.path.normpath(member.name))

                untracked = pre_existing - extracted
                if untracked:
                    if clear_untracked:
                        ScaleoutLogger().info(f"Clearing {len(untracked)} untracked file(s) from {self.pkg_path}")
                        _remove_files(self.pkg_path, untracked)
                    else:
                        ScaleoutLogger().warning(f"{len(untracked)} pre-existing file(s) in {self.pkg_path} are not part of the new package")

                ScaleoutLogger().info(f"Successfully extracted compute package content in {self.pkg_path}")
                return self._find_target_path(self.pkg_path)
            else:
                return None
        except Exception as e:
            ScaleoutLogger().error(f"Error extracting files: {e}")
            return None

    def _find_target_path(self, path) -> Optional[str]:
        for root, _, files in os.walk(os.path.join(path, "")):
            if self._target_name in files:
                ScaleoutLogger().info(f"Found {self._target_name} file in {root}")
                return root
        ScaleoutLogger().error(f"No {self._target_name} file found in {path}!")
        return None

    def load_local_compute_package(self, pkg_path) -> bool:
        """Initialize the local compute package."""
        path = self._find_target_path(pkg_path)
        if not path:
            ScaleoutLogger().error(f"Could not find {self._target_name} in the provided package path.")
            return False

        ScaleoutLogger().info(f"Using compute package at: {path}")
        self._target_path = path
        if not self._load_scaleoutyaml():
            ScaleoutLogger().error("Failed to load scaleout.yaml configuration file.")
            self._target_path = None
            return False

        if not self.init_runtime():
            ScaleoutLogger().error("Failed to initialize runtime package")
            return False
        return True

    def load_remote_compute_package(
        self,
        url: str,
        token: str,
        pkg_name: Optional[str] = None,
        validate: bool = True,
        clear_untracked: bool = False,
    ) -> bool:
        """Initialize the remote compute package.

        :param clear_untracked: If True, remove files in the package directory
            that are not present in the freshly extracted archive.
        """
        do_download = True
        if pkg_name and os.path.exists(os.path.join(self.tar_path, pkg_name)):
            # Package already exists
            ScaleoutLogger().info(f"Compute package {pkg_name} already exists in {self.tar_path}.")
            self.pkg_name = pkg_name
            if validate:
                result = self.validate_compute_package(url, token)
                if not result:
                    ScaleoutLogger().warning("Already downloaded compute package failed validation.")
                else:
                    ScaleoutLogger().info("Already downloaded compute package passed validation.")
                    do_download = False
            else:
                ScaleoutLogger().info("Skipping validation of already downloaded compute package.")
                do_download = False

        if do_download:
            result = self._download_compute_package(url, token, pkg_name)
            if not result:
                ScaleoutLogger().error("Could not download compute package")
                return False

            if validate:
                result = self.validate_compute_package(url, token)
                if not result:
                    ScaleoutLogger().error("Could not validate compute package")
                    return False

        path = self._unpack_compute_package(clear_untracked=clear_untracked)

        if not path:
            ScaleoutLogger().error("Could not unpack compute package")
            return False

        ScaleoutLogger().info(f"Compute package unpacked to: {path}")
        self._target_path = path
        if not self._load_scaleoutyaml():
            ScaleoutLogger().error("Failed to load scaleout.yaml configuration file.")
            self._target_path = None
            return False

        if not self.init_runtime():
            ScaleoutLogger().error("Failed to initialize runtime package")
            return False

        return True

    def _load_scaleoutyaml(self):
        """Load the target configuration file."""
        ScaleoutLogger().info(f"Reading {self._target_name} configuration file.")
        self.config = read_yaml_file(os.path.join(self._target_path, self._target_name))
        if not self.config:
            ScaleoutLogger().error(f"Configuration file {os.path.join(self._target_path, self._target_name)} not found or is empty.")
            return False
        return True

    def run_startup(self, *args, **kwargs):
        raise NotImplementedError("The start method should be implemented in subclasses.")

    def init_runtime(self) -> bool:
        raise NotImplementedError("The init_runtime should be implemented in subclasses.")


def valid_env_for_restart():
    args_list = sys.argv
    if "client" in args_list and "start" in args_list:
        # Find the index of "client" and "start"
        try:
            client_idx = args_list.index("client")
            start_idx = args_list.index("start", client_idx)
            # Everything after "start" are the arguments to pass
            if client_idx != start_idx - 1:
                return False
        except ValueError:
            return False
    else:
        return False
    return True


def restart_with_venv_and_package(venv_path=None):
    """Restart the client."""
    # This method could be replace by letting a process manager handle the restart, i.e. a watchdog or supervisor.
    # The watchdog would monitor the client process and restart it if it exits unexpectedly
    # and start the client with the correct environment activated.
    if venv_path is None:
        venv_path = sys.prefix

    venv_python = get_executable_path_from_venv_path(venv_path)
    if not Path(venv_python).exists():
        ScaleoutLogger().error("Python interpretor does not exist: " + venv_python)
        raise RuntimeError(f"Python interpretor does not exist: {venv_python}")
    ScaleoutLogger().info(f"Restarting client with: {venv_python}")

    # TODO: Maybe we need to close open tcp connections and/or open file handles before restarting

    # Sanitize args to avoid shell injection and ensure safe usage
    # Use shlex.split to safely parse the command line arguments
    args_list = sys.argv
    if "client" in args_list and "start" in args_list:
        # Find the index of "client" and "start"
        try:
            client_idx = args_list.index("client")
            start_idx = args_list.index("start", client_idx)
            # Everything after "start" are the arguments to pass
            if client_idx != start_idx - 1:
                ScaleoutLogger().warning("Unexpected arguments between 'client' and 'start'. These will be ignored.")
            args_after_start_list = args_list[start_idx + 1 :]
        except ValueError:
            raise RuntimeError("Invalid command line arguments for restarting the client.")
    else:
        ScaleoutLogger().error("The command does not contain 'client' and 'start'. Cannot restart safely.")
        raise RuntimeError("Invalid command line arguments for restarting the client.")
    ScaleoutLogger().info(f"Current command line arguments: {' '.join(args_list)}")

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = venv_path
    venv_bin = os.path.dirname(venv_python)
    current_path = env.get("PATH", "")
    if current_path.split(os.pathsep)[0] != venv_bin:
        env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"
    env.pop("PYTHONHOME", None)

    ScaleoutLogger().info(f"Restarting with interpreter: {venv_python}")
    ScaleoutLogger().info("Restarting in 2 seconds...")
    time.sleep(2)
    os.execve(venv_python, [venv_python, "-m", "scaleout", "client", "start"] + args_after_start_list, env)  # noqa: S606
    # This line will never be reached, as os.execve replaces the current process with a new one.
