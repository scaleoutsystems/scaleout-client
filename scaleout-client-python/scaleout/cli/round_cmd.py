import click

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


@main.group(
    "round",
    help="Commands to list and inspect rounds.",
    invoke_without_command=True,
)
@click.pass_context
def round_cmd(ctx):
    """Round commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-s", "--session_id", required=False, help="Rounds in session with given session id")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@round_cmd.command("list")
@click.pass_context
def list_rounds(
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
    """List rounds.

    **Returns**

    - count: number of rounds
    - result: list of rounds
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    filter = {"round_config.session_id": session_id} if session_id else None
    result = call(client.get_rounds, n_max=n_max, filter=filter)
    return render_response(result, "rounds", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-id", "--id", required=True, help="Round ID")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@round_cmd.command("get")
@click.pass_context
def get_round(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get round.

    **Returns**

    - result: round with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_round, id)
    return render_response(result, "round", output_format=output_format, base_url=base_url)
