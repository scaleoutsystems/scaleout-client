import click

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


@main.group(
    "status",
    help="Commands to list and get status entries.",
    invoke_without_command=True,
)
@click.pass_context
def status_cmd(ctx):
    """Status commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-s", "--session_id", required=False, help="statuses with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@status_cmd.command("list")
@click.pass_context
def list_statuses(
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
    """List statuses.

    **Returns**

    - count: number of statuses
    - result: list of statuses
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_statuses, session_id=session_id, n_max=n_max)
    return render_response(result, "statuses", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Status ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@status_cmd.command("get")
@click.pass_context
def get_status(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get status.

    **Returns**

    - result: status with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_status, id)
    return render_response(result, "status", output_format=output_format, base_url=base_url)
