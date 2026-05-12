import click

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_response, process_response


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
@status_cmd.command("list")
@click.pass_context
def list_statuses(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None, output_format: str = "human"):
    """List statuses.

    **Returns**

    - count: number of statuses
    - result: list of statuses
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    query = {}
    if session_id:
        query["session_id"] = session_id

    response = get_response(base_url=base_url, endpoint="statuses/", query=query, token=token, headers=headers)
    return process_response(response, "statuses", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Status ID")
@status_cmd.command("get")
@click.pass_context
def get_status(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get status.

    **Returns**

    - result: status with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"statuses/{id}", query={}, token=token, headers={})
    return process_response(response, "status", output_format=output_format, base_url=base_url)
