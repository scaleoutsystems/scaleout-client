"""Run several edge clients concurrently from a single shared enrollment token.

Each client runs in its own thread. Because no client id is pinned, every client
generates its own id and enrolls independently against the same enrollment token,
so the server records N distinct enrolled clients — each with its own short-lived
access token and 30-day refresh token, refreshed automatically in the background.

This mirrors what ``scaleout client start --token <enrollment-token>`` does, run
N times in parallel: the runtime detects the enrollment token and enrolls
transparently before connecting to the combiner.

Prerequisites
-------------
* A running Scaleout deployment (API + combiner).
* A compute package available to the clients — either the session's default
  remote package (``SCALEOUT_PACKAGE=remote``, the default; set
  ``SCALEOUT_PACKAGE_NAME`` only to pin a specific one) or a local ``client/``
  directory next to this script (``SCALEOUT_PACKAGE=local``).
* An enrollment token, e.g.::

      scaleout login https://your-deployment
      export SCALEOUT_ENROLLMENT_TOKEN="$(scaleout client create-enrollment-token --name fleet --expires-in 1h)"

Usage
-----
::

    export SCALEOUT_API_URL="https://your-deployment"
    export SCALEOUT_ENROLLMENT_TOKEN="<enrollment-token>"
    # optional: pin a specific uploaded package; omit to use the session's default
    # export SCALEOUT_PACKAGE_NAME="<uploaded-package-name>"
    python run_clients.py --clients 3

Stop with Ctrl+C.
"""

import argparse
import logging
import os
import threading

from scaleout.client.connect import ClientOptions
from scaleout.client.importer_client import ImporterClient
from scaleoututil.logging import ScaleoutLogger


def run_client(index: int, api_url: str, enrollment_token: str, package: str, package_name: str | None) -> None:
    """Enroll and start a single edge client. Blocks until the client stops."""
    logger = ScaleoutLogger()
    name = f"edge-{index}"
    logger.info(f"[{name}] enrolling and starting...")

    # client_id is left unset so ClientOptions mints a fresh one; the runtime enrolls
    # this id against the shared enrollment token when it connects.
    client_options = ClientOptions(name=name, package=package)

    client = ImporterClient(
        api_url=api_url,
        client_obj=client_options,
        # An enrollment token handed to the runtime is exchanged for client tokens
        # transparently before the client connects.
        refresh_token=enrollment_token,
        package_name=package_name,
    )

    try:
        client.start()  # blocks: connect -> load package -> run task loop
    except Exception as exc:  # keep one client's failure from killing the others
        logger.error(f"[{name}] exited with error: {exc}")
    else:
        logger.info(f"[{name}] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N edge clients from one shared enrollment token.")
    parser.add_argument("--clients", type=int, default=int(os.environ.get("SCALEOUT_NUM_CLIENTS", "3")), help="Number of concurrent clients.")
    args = parser.parse_args()

    api_url = os.environ.get("SCALEOUT_API_URL")
    enrollment_token = os.environ.get("SCALEOUT_ENROLLMENT_TOKEN")
    package = os.environ.get("SCALEOUT_PACKAGE", "remote")
    package_name = os.environ.get("SCALEOUT_PACKAGE_NAME")

    missing = [k for k, v in {"SCALEOUT_API_URL": api_url, "SCALEOUT_ENROLLMENT_TOKEN": enrollment_token}.items() if not v]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")

    # Libraries default to WARNING with no console handler; opt in to INFO so the
    # enrollment/connect/training progress of each client is actually visible.
    logger = ScaleoutLogger()
    logger.enable_console_logging(logging.INFO)
    # package_name is optional: when unset, each client downloads the session's
    # default/active compute package.
    logger.info(f"Starting {args.clients} clients against {api_url} (package={package}, package_name={package_name or 'default'}).")

    threads = [
        threading.Thread(
            target=run_client,
            args=(i, api_url, enrollment_token, package, package_name),
            name=f"edge-{i}",
            daemon=True,
        )
        for i in range(args.clients)
    ]
    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            # Timeout keeps the main thread responsive to Ctrl+C (join(None) would block signals).
            while thread.is_alive():
                thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted — clients are daemon threads and will be torn down on exit.")


if __name__ == "__main__":
    main()
