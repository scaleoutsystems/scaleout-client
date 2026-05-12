"""Package commands for the CLI."""

import fnmatch
import os
import sys
import tarfile

import click

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_api_url, get_response, get_scheme_token, process_response
from scaleout.cli.upload_util import perform_chunked_upload
import requests


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
@package_cmd.command("list")
@click.pass_context
def list_packages(_: click.Context, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human") -> None:
    """Return a list of packages.

    **Returns**

    - count: number of packages
    - result: list of packages
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(base_url=base_url, endpoint="packages/", query={}, token=token, headers=headers)
    return process_response(response, "packages", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Package ID")
@package_cmd.command("get")
@click.pass_context
def get_package(_: click.Context, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human") -> None:
    """Return a package with given id.

    **Returns**

    - result: package with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"packages/{id}", query={}, token=token, headers={})
    return process_response(response, "package", output_format=output_format, base_url=base_url)


@package_cmd.command("set-active")
@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the package file")
@click.option("-d", "--description", required=False, help="Description of the package")
@click.option("-n", "--name", required=True, help="Name of the package")
@click.option("--helper", required=False, default="numpyhelper", help="Helper to use for the package")
@click.pass_context
def set_active(
    _: click.Context, protocol: str, host: str, port: str, token: str = None, file: str = None, description: str = None, name: str = None, helper: str = None
) -> None:
    """Set a package as active.

    **Returns**

    - result: package with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}
    _token = get_scheme_token(token=token)
    if _token:
        headers = {"Authorization": _token}

    url = get_api_url(base_url, "packages/")
    file_token = perform_chunked_upload(base_url, token, file, headers)
    response = requests.post(
        url,
        data={"helper": helper, "name": name, "description": description, "file_token": file_token, "file_name": os.path.basename(file)},
        headers=headers,
        verify=False,
    )
    if 200 <= response.status_code <= 204:
        click.secho("Package set as active successfully.", fg="green")
    else:
        click.secho(f"Failed to set package as active. Status code: {response.status_code}, Response: {response.text}", fg="red")
