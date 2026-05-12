import os
import tarfile
from typing import Optional

import requests

from scaleoututil.config import SCALEOUT_ARCHIVE_DIR, SCALEOUT_AUTH_SCHEME, SCALEOUT_CONNECT_API_SECURE, SCALEOUT_PACKAGE_EXTRACT_DIR
from scaleoututil.logging import ScaleoutLogger
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

    def _unpack_compute_package(self) -> Optional[str]:
        """Unpack the compute package.

        :return: Tuple containing a boolean indicating success and the path to the unpacked package.
        :rtype: Tuple[bool, str]
        """
        if not self.pkg_name:
            ScaleoutLogger().error("Failed to unpack compute package, no pkg_name set. Has the reducer been configured with a compute package?")
            return False, ""

        try:
            if self.pkg_name.endswith(("tar.gz", ".tgz", "tar.bz2")):
                tar_path = os.path.join(self.tar_path, self.pkg_name)
                with tarfile.open(tar_path, "r:*") as f:
                    for member in f.getmembers():
                        f.extract(member, self.pkg_path)
                ScaleoutLogger().info(f"Successfully extracted compute package content in {self.pkg_path}")
                return self.find_target_path(self.pkg_path)
            else:
                return None
        except Exception as e:
            ScaleoutLogger().error(f"Error extracting files: {e}")
            return None

    def find_target_path(self, path) -> Optional[str]:
        for root, _, files in os.walk(os.path.join(path, "")):
            if self._target_name in files:
                ScaleoutLogger().info(f"Found {self._target_name} file in {root}")
                return root
        ScaleoutLogger().error(f"No {self._target_name} file found in {path}!")
        return None

    def load_local_compute_package(self, pkg_path) -> bool:
        """Initialize the local compute package."""
        path = self.find_target_path(pkg_path)
        if not path:
            ScaleoutLogger().error(f"Could not find {self._target_name} in the provided package path.")
            return False

        ScaleoutLogger().info(f"Using compute package at: {path}")
        self._target_path = path
        if not self._load_fednyaml():
            ScaleoutLogger().error("Failed to load scaleout.yaml configuration file.")
            self._target_path = None
            return False
        return True

    def load_remote_compute_package(self, url: str, token: str, pkg_name: Optional[str] = None, validate: bool = True) -> bool:
        """Initialize the remote compute package."""
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

        path = self._unpack_compute_package()

        if not path:
            ScaleoutLogger().error("Could not unpack compute package")
            return False

        ScaleoutLogger().info(f"Compute package unpacked to: {path}")
        self._target_path = path
        if not self._load_fednyaml():
            ScaleoutLogger().error("Failed to load scaleout.yaml configuration file.")
            self._target_path = None
            return False
        return True

    def _load_fednyaml(self):
        """Load the target configuration file."""
        ScaleoutLogger().info(f"Reading {self._target_name} configuration file.")
        self.config = read_yaml_file(os.path.join(self._target_path, self._target_name))
        if not self.config:
            ScaleoutLogger().error(f"Configuration file {os.path.join(self._target_path, self._target_name)} not found or is empty.")
            return False
        return True

    def run_startup(self, *args, **kwargs):
        raise NotImplementedError("The start method should be implemented in subclasses.")
