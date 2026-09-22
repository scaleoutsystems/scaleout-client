import sys

import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


@main.group(
    "session",
    help="Commands to list sessions, inspect a session, and start a new training session.",
    invoke_without_command=True,
)
@click.pass_context
def session_cmd(ctx):
    """Session commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@session_cmd.command("list")
@click.pass_context
def list_sessions(ctx, *, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human", no_verify_tls: bool = False):
    """List sessions."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_sessions, n_max=n_max)
    return render_response(result, "sessions", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Session ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@session_cmd.command("get")
@click.pass_context
def get_session(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get session by id."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_session, id)
    return render_response(result, "session", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@session_cmd.command("stop")
@click.pass_context
def stop_session(ctx, *, protocol: str, host: str, port: str, token: str = None, no_verify_tls: bool = False):
    """Stop a session."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    try:
        result = client.stop_current_command()
    except RuntimeError as e:
        click.secho(f"Failed to stop session: {e}", fg="red")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.secho(f"Could not connect to the controller API at {base_url}: {e}", fg="red")
        sys.exit(1)
    click.secho(f"Control response: {result.get('message', 'Session stopped.')}", fg="green")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-n", "--name", required=False, help="Name of the session")
@click.option("-a", "--aggregator", required=False, default="fedavg", help="The aggregator plugin to use")
@click.option("-ak", "--aggregator_kwargs", required=False, type=dict, help="Aggregator keyword arguments")
@click.option("-m", "--model_id", required=False, help="The id of the initial model")
@click.option("-rt", "--round_timeout", required=False, default=180, type=int, help="The round timeout to use in seconds")
@click.option("-r", "--rounds", required=False, default=5, type=int, help="The number of rounds to perform")
@click.option("-rb", "--round_buffer_size", required=False, default=-1, type=int, help="The round buffer size to use")
@click.option("-d", "--delete_models", required=False, default=True, type=bool, help="Whether to delete models after each round at combiner (save storage)")
@click.option("-v", "--validate", required=False, default=True, type=bool, help="Whether to validate the model after each round")
@click.option("-hp", "--helper", required=False, help="The helper type to use")
@click.option("-mc", "--min_clients", required=False, default=1, type=int, help="The minimum number of clients required")
@click.option("-rc", "--requested_clients", required=False, default=8, type=int, help="The requested number of clients")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@session_cmd.command("start")
@click.pass_context
def start_session(
    ctx,
    *,
    protocol: str,
    host: str,
    port: str,
    token: str,
    name: str = None,
    aggregator: str = "fedavg",
    aggregator_kwargs: dict = None,
    model_id: str = None,
    round_timeout: int = 180,
    rounds: int = 5,
    round_buffer_size: int = -1,
    delete_models: bool = True,
    validate: bool = True,
    helper: str = None,
    min_clients: int = 1,
    requested_clients: int = 8,
    no_verify_tls: bool = False,
):
    """Start a new session."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(
        client.start_session,
        name=name,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        model_id=model_id,
        round_timeout=round_timeout,
        rounds=rounds,
        round_buffer_size=round_buffer_size,
        delete_models=delete_models,
        validate=validate,
        helper=helper,
        min_clients=min_clients,
        requested_clients=requested_clients,
    )
    return render_response(result, "session", base_url=base_url)
