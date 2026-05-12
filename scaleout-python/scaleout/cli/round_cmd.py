import click

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_response, process_response


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
@round_cmd.command("list")
@click.pass_context
def list_rounds(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None, output_format: str = "human"):
    """List rounds.

    **Returns**

    - count: number of rounds
    - result: list of rounds
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    query = {}
    if session_id:
        query["round_config.session_id"] = session_id

    response = get_response(base_url=base_url, endpoint="rounds/", query=query, token=token, headers=headers)
    return process_response(response, "rounds", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-id", "--id", required=True, help="Round ID")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-t", "--token", required=False, help="Authentication token")
@round_cmd.command("get")
@click.pass_context
def get_round(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get round.

    **Returns**

    - result: round with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"rounds/{id}", query={}, token=token, headers={})
    return process_response(response, "round", output_format=output_format, base_url=base_url)
