"""Launch N continual-learning clients programmatically against a Scaleout Edge API.

Thin CLI wrapper around :func:`scaleoututil.launchers.launch_clients`. The
launcher hands a single enrollment token to N ``scaleout client start``
subprocesses (each exchanges it for its own client credentials at startup),
streams their logs, and serializes train/validate compute via a shared file
lock; the client compute package itself does not need to know about any of that.

Mint an enrollment token first with::

    scaleout client create-enrollment-token --name continual-learning

Then::

    python run.py --api-url <URL> --enrollment-token <TOKEN> -n 4
"""

import argparse
import os
import sys

from scaleoututil.launchers import launch_clients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api-url", required=True, help="Scaleout API URL (e.g. http://localhost)")
    parser.add_argument("--api-port", type=int, default=None, help="Scaleout API port (default: derived from URL/scheme)")
    parser.add_argument("--enrollment-token", required=True, help="Enrollment JWT (from 'scaleout client create-enrollment-token'); shared by all clients")
    parser.add_argument("-n", "--num-clients", type=int, required=True, help="Number of client subprocesses to launch")
    parser.add_argument("--client-name-prefix", default="cl-client", help="Prefix for the --name passed to each client")
    parser.add_argument("--remote-package", action="store_true", help="Start each client with --remote-package (download compute package from controller)")
    parser.add_argument(
        "-c",
        "--preferred-combiner",
        default=None,
        help=(
            "Name of the combiner every client should attach to (e.g. combiner-1). "
            "Required on deployments with more than one combiner: a round is satisfied per "
            "combiner, so clients spread across several of them can never meet clients_required."
        ),
    )
    parser.add_argument(
        "--disable-managed-env",
        action="store_true",
        help="Run clients in the current Python environment instead of a managed virtualenv built from python_env.yaml.",
    )
    parser.add_argument(
        "--launch-stagger",
        type=float,
        default=0.2,
        help="Seconds to wait between launching clients, to avoid stampeding the controller on connect (default: 0.2)",
    )
    args = parser.parse_args()

    extra_client_args = []
    if args.preferred_combiner:
        extra_client_args += ["--preferred-combiner", args.preferred_combiner]
    if args.disable_managed_env:
        extra_client_args.append("--disable-managed-env")

    return launch_clients(
        api_url=args.api_url,
        api_port=args.api_port,
        enrollment_token=args.enrollment_token,
        num_clients=args.num_clients,
        client_name_prefix=args.client_name_prefix,
        remote_package=args.remote_package,
        project_dir=os.path.dirname(os.path.abspath(__file__)),
        serialize_compute=True,
        stagger_seconds=args.launch_stagger,
        extra_client_args=extra_client_args,
    )


if __name__ == "__main__":
    sys.exit(main())
