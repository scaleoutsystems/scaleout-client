"""Launchers for multi-client Scaleout Edge examples.

Right now this module exposes a single helper, :func:`launch_clients`, that
turns "spin up N clients against my Scaleout Edge controller" into one
function call. It:

* takes a single admin-issued *enrollment token* and hands the same token to
  every client; ``scaleout client start`` exchanges it for that client's own
  credentials at startup, so one enrollment token enrolls all ``N`` siblings
  without any per-client token minting in the parent;
* spawns ``N`` ``scaleout client start`` subprocesses, one per client, with
  ``CLIENT_NUMBER`` and ``TOTAL_N_CLIENTS`` exported in each child's env;
* optionally creates a shared file-lock path and exports it as
  ``SCALEOUT_CLIENT_COMPUTE_LOCK`` so the SDK transparently serializes
  train/validate compute across siblings (useful when all clients share one
  CPU/GPU);
* streams each child's stdout/stderr back to the parent with a
  ``[client i] `` prefix and propagates ``SIGINT`` / ``SIGTERM``.

Compared to running ``scaleout client start`` by hand in N terminals this is
most useful for examples, benchmarks, and any "I want N federated clients on
one box" workflow.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

__all__ = ["launch_clients"]


def _stream_with_prefix(prefix: str, src) -> None:
    """Forward bytes from ``src`` to stdout, prefixing each line."""
    for raw in iter(src.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - cosmetic decode fallback
            line = repr(raw)
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()


def launch_clients(
    *,
    api_url: str,
    enrollment_token: str,
    num_clients: int,
    api_port: Optional[int] = None,
    client_name_prefix: str = "client",
    remote_package: bool = False,
    project_dir: Optional[str] = None,
    serialize_compute: bool = False,
    stagger_seconds: float = 0.0,
    scaleout_bin: str = "scaleout",
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """Spawn ``num_clients`` ``scaleout client start`` subprocesses sharing one enrollment token.

    :param api_url: Scaleout API URL (e.g. ``http://localhost`` or
        ``https://api.example.com``). Passed as ``--api-url`` to each
        subprocess.
    :param enrollment_token: Admin-issued enrollment JWT (created via
        ``scaleout client create-enrollment-token``). The same token is given
        to every client; each ``scaleout client start`` exchanges it for its
        own client credentials at startup, so a single enrollment token
        enrolls all ``num_clients`` siblings.
    :param num_clients: Number of client subprocesses to launch (``>= 1``).
    :param api_port: Optional API port. Defaults to whatever ``api_url``
        implies (80 for http, 443 for https).
    :param client_name_prefix: Each client is started with
        ``--name {client_name_prefix}-{i}``.
    :param remote_package: If ``True``, each client is started with
        ``--remote-package``; the compute package is downloaded from the
        controller. If ``False`` (default), each client reads the local
        ``client/scaleout.yaml`` from ``project_dir``.
    :param project_dir: Working directory for each subprocess. Defaults to
        the caller's CWD. ``scaleout client start`` resolves the local
        compute package relative to this directory when
        ``remote_package=False``.
    :param serialize_compute: If ``True``, all subprocesses share a file
        lock and the SDK wraps user train/validate callbacks in
        ``with FileLock(path):`` so only one client computes at a time.
        Useful when N clients share one GPU/CPU and you want peak memory
        bounded. Implemented entirely outside the user's compute package;
        the package itself does not need to know about the lock.
    :param stagger_seconds: Seconds to wait between spawning consecutive
        subprocesses (default ``0.0`` -- spawn back-to-back). A small value
        spreads out the initial ``connect_to_api`` calls so a large fleet
        doesn't hit the controller all at once (thundering herd). The client
        also retries connect with backoff, so this is just an extra
        smoothing knob, not a correctness requirement.
    :param scaleout_bin: Path to the ``scaleout`` CLI executable.
    :param extra_env: Extra env vars to set on every subprocess. Merged
        after ``CLIENT_NUMBER``/``TOTAL_N_CLIENTS``/``SCALEOUT_CLIENT_COMPUTE_LOCK``
        so the caller can override them if needed.
    :return: ``0`` if all subprocesses exited cleanly, otherwise the exit
        code of the first failing subprocess.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be >= 1")
    if not enrollment_token or not enrollment_token.strip():
        raise ValueError("enrollment_token must be a non-empty enrollment JWT")

    lock_path: Optional[str] = None
    if serialize_compute:
        lock_dir = tempfile.mkdtemp(prefix="scaleout-clients-")
        lock_path = os.path.join(lock_dir, "compute.lock")
        print(f"Serializing train/validate compute via {lock_path}")

    procs: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []

    def cleanup(signum=None, frame=None):  # noqa: ARG001 - signal handler
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001 - best effort
                    pass

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    cwd = project_dir or os.getcwd()

    for i in range(num_clients):
        env = {
            **os.environ,
            "CLIENT_NUMBER": str(i),
            "TOTAL_N_CLIENTS": str(num_clients),
        }
        if lock_path:
            env["SCALEOUT_CLIENT_COMPUTE_LOCK"] = lock_path
        if extra_env:
            env.update(extra_env)

        cmd = [
            scaleout_bin,
            "client",
            "start",
            "--api-url",
            str(api_url),
            "--token",
            enrollment_token,
            "--name",
            f"{client_name_prefix}-{i}",
        ]
        if api_port:
            cmd += ["--api-port", str(api_port)]
        if remote_package:
            cmd.append("--remote-package")

        printable = list(cmd)
        if "--token" in printable:
            tok_idx = printable.index("--token")
            if tok_idx + 1 < len(printable):
                printable[tok_idx + 1] = "<redacted>"
        print(f"[client {i}] launching: {' '.join(printable)}")

        p = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

        t = threading.Thread(
            target=_stream_with_prefix,
            args=(f"[client {i}] ", p.stdout),
            daemon=True,
        )
        t.start()
        threads.append(t)

        # Spread out the initial connect_to_api calls so a large fleet does not
        # stampede the controller. Skip the wait after the last spawn.
        if stagger_seconds > 0 and i < num_clients - 1:
            time.sleep(stagger_seconds)

    exit_code = 0
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            print(f"[client {i}] exited with code {rc}")
            exit_code = exit_code or rc

    for t in threads:
        t.join(timeout=2.0)

    return exit_code
