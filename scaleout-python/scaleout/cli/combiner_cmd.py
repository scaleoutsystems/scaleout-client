import click

from scaleout.cli.main import main
from scaleout.cli.shared import get_response, process_response, complement_with_context, patch_request


@main.group(
    "combiner",
    help="Commands to list and inspect combiners.",
    invoke_without_command=True,
)
@click.pass_context
def combiner_cmd(ctx):
    """Combiner commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Combiner ID")
@click.argument("public_hostname", required=True)
@combiner_cmd.command("set-public-hostname")
@click.pass_context
def set_public_hostname(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, public_hostname: str = None):
    """Set the public host of a combiner."""
    base_url, token = complement_with_context(protocol, host, port, token)
    data = {"public_hostname": public_hostname}
    response = patch_request(base_url=base_url, endpoint=f"combiners/{id}", query={}, token=token, data=data, headers={})
    if response.status_code == 200:
        click.secho("Public host updated successfully.", fg="green")
    else:
        click.secho(f"Failed to update public host. Status code: {response.status_code}", fg="red")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@combiner_cmd.command("list")
@click.pass_context
def list_combiners(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human"):
    """List combiners.

    **Returns**

    - count: number of combiners
    - result: list of combiners
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(base_url=base_url, endpoint="combiners/", query={}, token=token, headers=headers)
    return process_response(response, "combiners", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Combiner ID")
@combiner_cmd.command("get")
@click.pass_context
def get_combiner(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human"):
    """Get combiner.

    **Returns**

    - result: combiner with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"combiners/{id}", query={}, token=token, headers={})
    return process_response(response, "combiner", output_format=output_format, base_url=base_url)
