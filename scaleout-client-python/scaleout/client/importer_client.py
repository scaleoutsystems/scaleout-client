"""Client module for handling client operations in the Scaleout network."""

import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, Callable
from scaleout.client.connect import ClientOptions, get_url
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.edge_client import ConnectToApiResult, EdgeClient, GrpcConnectionOptions
from scaleout.client.importer_package_runtime import ImporterPackageRuntime, get_compute_package_dir_path


class ImporterClient:
    """Client for interacting with the Scaleout network."""

    def __init__(
        self,
        api_url: str,
        client_obj: ClientOptions,
        combiner_host: Optional[str] = None,
        combiner_port: Optional[int] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        package_checksum: Optional[str] = None,
        helper_type: Optional[str] = None,
        startup_path: Optional[str] = None,
        managed_env: Optional[bool] = False,
        token_refresh_callback: Optional[Callable[[str, str, datetime], None]] = None,
    ) -> None:
        """Initialize the Client."""
        self.api_url = api_url
        self.combiner_host = combiner_host
        self.combiner_port = combiner_port
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_obj = client_obj
        self.package_checksum = package_checksum
        self.helper_type = helper_type

        package_path, archive_path = get_compute_package_dir_path()
        self.package_runtime = ImporterPackageRuntime(package_path, archive_path)

        self.managed_env = managed_env
        self.fedn_api_url = get_url(self.api_url)

        self.edge_client: EdgeClient = EdgeClient()

        self.helper = None
        self.startup_path = startup_path

        self.token_refresh_callback = token_refresh_callback

    def _connect_to_api(self) -> Tuple[bool, Optional[dict]]:
        """Connect to the API and handle retries."""
        result = None
        response = None

        while not result or result == ConnectToApiResult.ComputePackageMissing:
            if result == ConnectToApiResult.ComputePackageMissing:
                ScaleoutLogger().info("Retrying in 3 seconds")
                time.sleep(3)
            result, response = self.edge_client.connect_to_api(
                url=self.fedn_api_url, json=self.client_obj.to_json(), token=self.refresh_token, token_refresh_callback=self.token_refresh_callback
            )

        if result == ConnectToApiResult.Assigned:
            return True, response

        return False, None

    def start(self) -> None:
        """Start the client."""
        if self.combiner_host and self.combiner_port:
            combiner_config = GrpcConnectionOptions(host=self.combiner_host, port=self.combiner_port)
        else:
            success, combiner_config = self._connect_to_api()
            if not success:
                return
        if self.client_obj.package == "remote":
            # Get access token from edge_client's TokenManager
            access_token = self.edge_client.get_access_token() if self.edge_client else None
            success = self.package_runtime.load_remote_compute_package(url=self.fedn_api_url, token=access_token)
            if not success:
                return
        else:
            success = self.package_runtime.load_local_compute_package(os.path.join(os.getcwd(), "client"))
            if not success:
                return

        self.edge_client.package_path = self.package_runtime.package_path

        if self.managed_env:
            ScaleoutLogger().info("Using managed environment")
            try:
                self.package_runtime.update_runtime_env()
            except Exception as e:
                ScaleoutLogger().error(f"Failed to manage runtime environment: {e}")
                ScaleoutLogger().error("Client exiting...")
                return
            if self.package_runtime.requires_restart:
                self.__restart_client()
            else:
                ScaleoutLogger().info("Managed environment is active and verified.")
        else:
            ScaleoutLogger().info("Managed environment is disabled, running in current environment.")

        success = self.edge_client.init_grpchandler(
            config=combiner_config, token=self.refresh_token, url=self.fedn_api_url, token_refresh_callback=self.token_refresh_callback
        )
        if not success:
            return

        success = self.package_runtime.run_startup(self.edge_client)
        if not success:
            return

        self.edge_client.run()

    def __restart_client(self) -> None:
        """Restart the client."""
        # This method could be replace by letting a process manager handle the restart, i.e. a watchdog or supervisor.
        # The watchdog would monitor the client process and restart it if it exits unexpectedly
        # and start the client with the correct environment activated.
        ScaleoutLogger().info("Restarting client with managed environment.")

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
        ScaleoutLogger().info("Restarting in 2 seconds...")
        time.sleep(2)
        os.execv(sys.executable, [sys.executable, "-m", "scaleout", "client", "start"] + args_after_start_list)  # noqa: S606
        # This line will never be reached, as os.execv replaces the current process with a new one.
