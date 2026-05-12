"""Client module for handling client operations in the Scaleout network."""

import json
import os
import tempfile
import time
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from scaleout.client.connect import ClientOptions, get_url
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.dispatcher_package_runtime import DispatcherPackageRuntime
from scaleout.client.edge_client import ConnectToApiResult, EdgeClient, GrpcConnectionOptions
from scaleout.client.package_runtime import get_compute_package_dir_path
from scaleoututil.helpers.helpers import get_helper, save_metadata
from scaleoututil.utils.model import ScaleoutModel


def get_tmp_path():
    """Return a temporary output path compatible with save_model, load_model."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return path


class DispatcherClient:
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
        self.token_refresh_callback = token_refresh_callback

        package_path, archive_path = get_compute_package_dir_path()
        self._package_runtime = DispatcherPackageRuntime(package_path, archive_path)

        self.edge_client: EdgeClient = EdgeClient()

        self.fedn_api_url = get_url(self.api_url)
        self.helper = None

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
            result, combiner_config = self._connect_to_api()
            if not result:
                return
        if self.client_obj.package == "remote":
            # Get access token from edge_client's TokenManager
            access_token = self.edge_client.get_access_token() if self.edge_client else None
            result = self._package_runtime.load_remote_compute_package(url=self.fedn_api_url, token=access_token)
            if not result:
                return
        else:
            result = self._package_runtime.load_local_compute_package(os.path.join(os.getcwd(), "client"))
            if not result:
                return

        result = self._package_runtime.run_startup()
        if not result:
            return

        self.set_helper(combiner_config)

        result = self.edge_client.init_grpchandler(
            config=combiner_config, token=self.refresh_token, url=self.fedn_api_url, token_refresh_callback=self.token_refresh_callback
        )
        if not result:
            return

        ScaleoutLogger().info("-----------------------------")

        self.edge_client.set_train_callback(self.on_train)
        self.edge_client.set_validate_callback(self.on_validation)

        self.edge_client.set_name(self.client_obj.name)
        self.edge_client.set_client_id(self.client_obj.client_id)

        self.edge_client.run()

    def set_helper(self, response: Optional[GrpcConnectionOptions] = None) -> None:
        """Set the helper based on the response or default."""
        helper_type = response.helper_type if response else None
        helper_type_to_use = self.helper_type or helper_type or "numpyhelper"
        ScaleoutLogger().info(f"Setting helper to: {helper_type_to_use}")
        self.helper = get_helper(helper_type_to_use)

    def on_train(self, in_model: ScaleoutModel, client_settings: dict) -> Tuple[Optional[ScaleoutModel], dict]:
        """Handle the training callback."""
        return self._process_training_request(in_model, client_settings)

    def on_validation(self, in_model: ScaleoutModel) -> Optional[dict]:
        """Handle the validation callback."""
        return self._process_validation_request(in_model)

    def _process_training_request(self, in_model: ScaleoutModel, client_settings: dict) -> Tuple[Optional[ScaleoutModel], dict]:
        """Process a training (model update) request."""
        try:
            meta = {}
            inpath = self.helper.get_tmp_path()

            in_model.save_to_file(inpath)

            save_metadata(metadata=client_settings, filename=inpath)
            outpath = self.helper.get_tmp_path()
            tic = time.time()

            self._package_runtime.dispatcher.run_cmd(f"train {inpath} {outpath}")
            meta["exec_training"] = time.time() - tic

            out_model = ScaleoutModel.from_file(outpath)

            with open(outpath + "-metadata", "r") as fh:
                training_metadata = json.loads(fh.read())

            ScaleoutLogger().info(f"SETTING Training metadata: {training_metadata}")
            meta["training_metadata"] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath + "-metadata")

        except Exception as e:
            ScaleoutLogger().error(f"Could not process training request due to error: {e}")
            out_model = None
            meta = {"status": "failed", "error": str(e)}

        return out_model, meta

    def _process_validation_request(self, in_model: ScaleoutModel) -> Dict:
        """Process a validation request."""
        try:
            inpath = self.helper.get_tmp_path()

            in_model.save_to_file(inpath)

            outpath = get_tmp_path()
            self._package_runtime.dispatcher.run_cmd(f"validate {inpath} {outpath}")

            with open(outpath, "r") as fh:
                metrics = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            ScaleoutLogger().warning(f"Validation failed with exception {e}")
            metrics = None

        return metrics
