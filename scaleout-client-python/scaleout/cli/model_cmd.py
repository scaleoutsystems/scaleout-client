import sys

import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


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
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@model_cmd.command("list")
@click.pass_context
def list_models(
    ctx,
    *,
    protocol: str,
    host: str,
    port: str,
    token: str = None,
    session_id: str = None,
    n_max: int = None,
    output_format: str = "human",
    no_verify_tls: bool = False,
):
    """List models."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_models, session_id=session_id, n_max=n_max)
    return render_response(result, "models", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Model ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@model_cmd.command("get")
@click.pass_context
def get_model(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get model by id."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_model, id)
    return render_response(result, "model", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the model file")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@model_cmd.command("set-active")
@click.pass_context
def set_active_model(ctx, *, protocol: str, host: str, port: str, token: str, file: str, no_verify_tls: bool = False):
    """Set the initial model and upload to model repository."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    try:
        result = client.set_active_model(file)
    except (FileNotFoundError, RuntimeError) as e:
        click.secho(f"Upload failed: {e}", fg="red")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.secho(f"Could not connect to the controller API at {base_url}: {e}", fg="red")
        sys.exit(1)
    return render_response(result, "model", base_url=base_url)
