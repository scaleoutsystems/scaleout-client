import sys

import click
import requests

from scaleout.cli.main import main
from scaleout.cli.shared import build_client, call, render_response


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
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@click.argument("public_hostname", required=True)
@combiner_cmd.command("set-public-hostname")
@click.pass_context
def set_public_hostname(
    ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, public_hostname: str = None, no_verify_tls: bool = False
):
    """Set the public host of a combiner."""
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    try:
        result = client.set_public_hostname(id, public_hostname)
    except RuntimeError as e:
        click.secho(f"Failed to set public hostname: {e}", fg="red")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.secho(f"Could not connect to the controller API at {base_url}: {e}", fg="red")
        sys.exit(1)
    click.secho(result.get("message", "Public hostname updated."), fg="green")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@combiner_cmd.command("list")
@click.pass_context
def list_combiners(
    ctx, *, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human", no_verify_tls: bool = False
):
    """List combiners.

    **Returns**

    - count: number of combiners
    - result: list of combiners
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_combiners, n_max=n_max)
    return render_response(result, "combiners", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Combiner ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@combiner_cmd.command("get")
@click.pass_context
def get_combiner(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get combiner.

    **Returns**

    - result: combiner with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_combiner, id)
    return render_response(result, "combiner", output_format=output_format, base_url=base_url)
