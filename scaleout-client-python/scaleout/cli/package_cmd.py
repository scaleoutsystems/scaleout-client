"""Package commands for the CLI."""

import fnmatch
import os
import sys
import tarfile

import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


def create_tar_with_ignore(path: str, output_path: str) -> None:
    """Create a tar archive from a directory with an ignore and scaleout.yaml file."""
    try:
        ignore_patterns = []
        ignore_file = os.path.join(path, ".scaleoutignore")
        if os.path.exists(ignore_file):
            # Read ignore patterns from .scaleoutignore file
            with open(ignore_file, "r") as f:
                ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        def is_ignored(file_path: str) -> bool:
            relative_path = os.path.relpath(file_path, path)
            return any(fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern) for pattern in ignore_patterns)

        with tarfile.open(output_path, "w:gz") as tar:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path):
                        tar.add(file_path, arcname=os.path.relpath(file_path, path))

        click.secho(f"Created tar archive: {output_path}")
    except FileNotFoundError as e:
        click.secho(f"File not found: {e}", fg="red")
    except PermissionError as e:
        click.secho(f"Permission denied: {e}", fg="red")
    except Exception as e:
        click.secho(f"An error occurred: {e}", fg="red")


@main.group(
    "package",
    help="Commands to create, list/inspect, and upload compute packages.",
    invoke_without_command=True,
)
@click.pass_context
def package_cmd(ctx: click.Context) -> None:
    """Package commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@package_cmd.command("create")
@click.option("-p", "--path", required=True, help="Path to package directory containing scaleout.yaml")
@click.option("-n", "--name", required=False, default="package.tgz", help="Name of package tarball")
@click.option("-o", "--output", required=False, default=os.getcwd(), help="Output directory for the generated tarball")
@click.pass_context
def create_cmd(_: click.Context, path: str, name: str, output: str) -> None:
    """Create compute package.

    Make a tar.gz archive of folder given by --path. The archive will be named --name and saved in --output.
    """
    try:
        path = os.path.abspath(path)
        output = os.path.abspath(output)
        yaml_file = os.path.join(path, "scaleout.yaml")
        if not os.path.exists(yaml_file):
            click.secho(f"Could not find scaleout.yaml in {path}", fg="red")
            sys.exit(-1)

        if not os.path.exists(output):
            click.secho(f"Output directory does not exist: {output}", fg="red")
            sys.exit(-1)

        tar_path = os.path.join(output, name)
        create_tar_with_ignore(path, tar_path)
    except Exception as e:
        click.secho(f"An error occurred: {e}", fg="red")
        sys.exit(-1)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@package_cmd.command("list")
@click.pass_context
def list_packages(
    _: click.Context, *, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human", no_verify_tls: bool = False
) -> None:
    """Return a list of packages.

    **Returns**

    - count: number of packages
    - result: list of packages
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_packages, n_max=n_max)
    return render_response(result, "packages", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Package ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@package_cmd.command("get")
@click.pass_context
def get_package(
    _: click.Context, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False
) -> None:
    """Return a package with given id.

    **Returns**

    - result: package with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_package, id)
    return render_response(result, "package", output_format=output_format, base_url=base_url)


@package_cmd.command("set-active")
@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the package file")
@click.option("-d", "--description", required=False, help="Description of the package")
@click.option("-n", "--name", required=True, help="Name of the package")
@click.option("--helper", required=False, default="numpyhelper", help="Helper to use for the package")
@click.option("--restart-clients", is_flag=True, default=False, help="Push the new package to connected clients and restart them.")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@click.pass_context
def set_active(
    _: click.Context,
    *,
    protocol: str,
    host: str,
    port: str,
    token: str = None,
    file: str = None,
    description: str = None,
    name: str = None,
    helper: str = None,
    restart_clients: bool = False,
    no_verify_tls: bool = False,
) -> None:
    """Set a package as active.

    **Returns**

    - result: package with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    try:
        result = client.set_active_package(path=file, helper=helper, name=name, description=description or "", restart_clients=restart_clients)
    except (FileNotFoundError, RuntimeError) as e:
        click.secho(f"Upload failed: {e}", fg="red")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.secho(f"Could not connect to the controller API at {base_url}: {e}", fg="red")
        sys.exit(1)
    return render_response(result, "package", base_url=base_url)
