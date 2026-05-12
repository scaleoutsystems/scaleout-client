import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_api_url, get_response, get_scheme_token, process_response


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
@session_cmd.command("list")
@click.pass_context
def list_sessions(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human"):
    """List sessions."""
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(base_url=base_url, endpoint="sessions/", query={}, token=token, headers=headers)
    return process_response(response, "sessions", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Session ID")
@session_cmd.command("get")
@click.pass_context
def get_session(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get session by id."""
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"sessions/{id}", query={}, token=token, headers={})
    return process_response(response, "session", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@session_cmd.command("stop")
@click.pass_context
def stop_session(ctx, protocol: str, host: str, port: str, token: str = None):
    """Stop a session."""
    base_url, token = complement_with_context(protocol, host, port, token)
    response = requests.post(
        get_api_url(base_url, "control/stop"),
        headers={"Authorization": get_scheme_token(token=token)},
        verify=False,
    )
    if response.status_code == 200:
        json = response.json()
        click.secho(f"Control response: {json['message']}", fg="green")
    else:
        click.secho("Failed to send stop signal", fg="red")


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
@session_cmd.command("start")
@click.pass_context
def start_session(
    ctx,
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
):
    """Start a new session."""
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}
    _token = get_scheme_token(token=token)
    if _token:
        headers = {"Authorization": _token}

    if model_id is None:
        model_query_headers = headers.copy()
        model_query_headers["X-Limit"] = "1"
        model_query_headers["X-Sort-Key"] = "committed_at"
        model_query_headers["X-Sort-Order"] = "desc"
        model_query_headers["allow_training"] = "true"

        url = get_api_url(base_url, "models/")
        response = requests.get(url, headers=model_query_headers)
        if response.status_code == 200:
            json = response.json()

            if "result" in json and len(json["result"]) > 0:
                model_id = json["result"][0]["model_id"]
            else:
                click.secho("No models found", fg="red")
                return
        else:
            click.secho(f"Failed to get active model: {response.json()}", fg="red")
            return

    if helper is None:
        url = get_api_url(base_url, "helpers/active")
        response = requests.get(url, headers=headers)
        if response.status_code == 400:
            helper = "numpyhelper"
        elif response.status_code == 200:
            helper = response.json()
        else:
            click.secho("An unexpected error occurred when getting the active helper", fg="red")
            return

    url = get_api_url(base_url, "sessions/")
    response = requests.post(
        url,
        json={
            "name": name,
            "session_config": {
                "aggregator": aggregator,
                "aggregator_kwargs": aggregator_kwargs,
                "rounds": rounds,
                "round_timeout": round_timeout,
                "buffer_size": round_buffer_size,
                "model_id": model_id,
                "delete_models_storage": delete_models,
                "clients_required": min_clients,
                "requested_clients": requested_clients,
                "validate": validate,
                "helper_type": helper,
                "server_functions": None,
            },
        },
        headers=headers,
        verify=False,
    )

    if response.status_code == 201:
        session_id = response.json()["session_id"]
        url = get_api_url(base_url, "sessions/start")
        try:
            response = requests.post(
                url,
                json={
                    "session_id": session_id,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                },
                headers=headers,
                verify=False,
            )
            response_json = response.json()
            response_json["session_id"] = session_id
            if response.status_code != 200:
                click.secho(f"Failed to start session: {response.json()}", fg="red")
            else:
                click.secho(f"Session started successfully: {response_json}", fg="green")
        except requests.exceptions.RequestException:
            click.secho(f"Failed to start session: {response.json()}", fg="red")
    else:
        click.secho(f"Failed to start session: {response.json()}", fg="red")
