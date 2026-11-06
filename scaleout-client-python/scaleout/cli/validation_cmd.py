import click

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


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
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@validation_cmd.command("list")
@click.pass_context
def list_validations(
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
    """List validations.

    **Returns**

    - count: number of validations
    - result: list of validations
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_validations, session_id=session_id, n_max=n_max)
    return render_response(result, "validations", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="validation ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@validation_cmd.command("get")
@click.pass_context
def get_validation(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get validation.

    **Returns**

    - result: validation with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_validation, id)
    return render_response(result, "validation", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--session-id", required=True, help="Session ID")
@click.option("--model-id", required=True, help="Model ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@validation_cmd.command("start")
@click.pass_context
def start_validation(ctx, *, protocol: str, host: str, port: str, token: str = None, session_id: str = None, model_id: str = None, no_verify_tls: bool = False):
    """Start a validation for given session ID and model ID."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.start_validation, session_id, model_id)
    return render_response(result, "validation", base_url=base_url)
