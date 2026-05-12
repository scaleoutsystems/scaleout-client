import click

import requests
from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_api_url, get_response, get_scheme_token, process_response


@main.group(
    "validation",
    help="Commands to list and get validation results.",
    invoke_without_command=True,
)
@click.pass_context
def validation_cmd(ctx):
    """Validation commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-s", "--session_id", required=False, help="validations in session with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@validation_cmd.command("list")
@click.pass_context
def list_validations(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None, output_format: str = "human"):
    """List validations.

    **Returns**

    - count: number of validations
    - result: list of validations
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    query = {}
    if session_id:
        query["session_id"] = session_id

    response = get_response(base_url=base_url, endpoint="validations/", query=query, token=token, headers=headers)
    return process_response(response, "validations", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="validation ID")
@validation_cmd.command("get")
@click.pass_context
def get_validation(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get validation.

    **Returns**

    - result: validation with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"validations/{id}", query={}, token=token, headers={})
    return process_response(response, "validation", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--session-id", required=True, help="Session ID")
@click.option("--model-id", required=True, help="Model ID")
@validation_cmd.command("start")
@click.pass_context
def start_validation(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, model_id: str = None):
    """Start a validation for given session ID and model ID."""
    base_url, token = complement_with_context(protocol, host, port, token)
    data = {
        "session_id": session_id,
        "model_id": model_id,
    }
    response = requests.post(
        get_api_url(base_url, "validations/start"),
        json=data,
        headers={"Authorization": get_scheme_token(token=token)},
        verify=False,
    )
    if response.status_code == 200:
        click.secho("Validation started successfully.", fg="green")
    else:
        click.secho(f"Failed to start validation: {response.json()}", fg="red")
