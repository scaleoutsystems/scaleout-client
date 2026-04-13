import sys

import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_api_url, get_response, get_scheme_token, process_response
from scaleout.cli.upload_util import perform_chunked_upload


@main.group(
    "model",
    help="Commands to list models, get a model by ID, and set/upload the active model.",
    invoke_without_command=True,
)
@click.pass_context
def model_cmd(ctx):
    """Model commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-s", "--session_id", required=False, help="models in session with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@model_cmd.command("list")
@click.pass_context
def list_models(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None, output_format: str = "human"):
    """List models."""
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    query = {}
    if session_id:
        query["session_id"] = session_id

    response = get_response(base_url=base_url, endpoint="models/", query=query, token=token, headers=headers)
    return process_response(response, "models", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Model ID")
@model_cmd.command("get")
@click.pass_context
def get_model(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get model by id."""
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"models/{id}", query={}, token=token, headers={})
    return process_response(response, "model", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the model file")
@model_cmd.command("set-active")
@click.pass_context
def set_active_model(ctx, protocol: str, host: str, port: str, token: str, file: str):
    """Set the initial model and upload to model repository."""
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}
    _token = get_scheme_token(token=token)
    if _token:
        headers = {"Authorization": _token}

    if file.endswith(".npz"):
        helper = "numpyhelper"
    elif file.endswith(".bin"):
        helper = "binaryhelper"
    else:
        click.secho("Unsupported file type. Only .npz and .bin files are supported.", fg="red")
        return

    try:
        # Set the active helper
        url = get_api_url(base_url, "helpers/active")
        response_helper = requests.put(url, json={"helper": helper}, headers=headers, verify=False)
        response_helper.raise_for_status()
        if response_helper.status_code >= 200 and response_helper.status_code <= 204:
            click.secho(f"Active helper set to: {helper}", fg="green")
        else:
            click.secho(f"Failed to set active helper: {response_helper.text}", fg="red")
            sys.exit(1)

        # Upload the model file
        url = get_api_url(base_url, "models/")
        file_token = perform_chunked_upload(base_url, token, file, headers)
        click.secho(f"Uploading model entry with helper: {helper} to {url}", fg="yellow")
        response_model = requests.post(url, data={"helper": helper, "file_token": file_token}, headers=headers, verify=False)
        response_model.raise_for_status()

        if response_model.status_code >= 200 and response_model.status_code <= 204:
            click.secho("Model uploaded successfully.", fg="green")
        else:
            click.secho(f"Failed to upload model: {response_model.text}", fg="red")
            sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        click.secho(f"Could not connect to the controller at {host}:{port}. Is it running? Error: {e}", fg="red")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.secho(f"Failed to set active model: {e}", fg="red")
        sys.exit(1)
