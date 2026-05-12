from scaleout.utils.dist import VERSION
import click
import logging


import requests
import webbrowser
import time
import sys
from urllib.parse import urljoin

from scaleout.cli.shared import set_context, get_all_contexts, get_active_context, switch_context, remove_context
from scaleoututil.logging import ScaleoutLogger

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=["-h", "--help"],
)


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Scaleout command-line interface for starting clients and interacting with API resources such as models, sessions and more.",
    epilog="Use 'scaleout COMMAND --help' for detailed help on a command.",
    no_args_is_help=True,
)
@click.version_option(VERSION)
@click.pass_context
def main(ctx):
    """:param ctx:"""
    # Enable console logging for CLI usage
    ScaleoutLogger().enable_console_logging(logging.INFO)
    ctx.obj = dict()


@main.command("login")
@click.argument("instance_url")
@click.pass_context
def login(ctx, instance_url):
    instance_url = instance_url.rstrip("/")  # normalize

    # ---- STEP 1: Start login session ----
    init_url = urljoin(instance_url + "/", "api/cli-auth/init")

    try:
        res = requests.post(init_url, timeout=10)
    except Exception as e:
        click.echo(f"❌ Failed to reach {instance_url}: {e}", err=True)
        sys.exit(1)

    if res.status_code != 200:
        click.echo(f"❌ Server error: {res.text}", err=True)
        sys.exit(1)

    data = res.json()
    request_id = data["requestId"]
    poll_token = data["pollToken"]
    login_url = data["loginUrl"]

    click.echo(f"👉 Opening browser for login:\n   {login_url}\n")
    webbrowser.open(login_url)

    # ---- STEP 2: Poll for login completion ----
    status_url = urljoin(instance_url + "/", "api/cli-auth/status")
    click.echo("⏳ Waiting for you to complete login...")

    start = time.time()
    timeout_seconds = 300  # 5 min expiration

    while True:
        if time.time() - start > timeout_seconds:
            click.echo("❌ Login timed out.", err=True)
            sys.exit(1)

        time.sleep(2)

        poll = requests.get(
            status_url,
            params={
                "requestId": request_id,
                "pollToken": poll_token,
            },
        )

        if poll.status_code != 200:
            continue  # try again

        status = poll.json()

        if status["status"] == "pending":
            continue

        if status["status"] == "error":
            click.echo(f"❌ Login failed: {status.get('message', 'unknown error')}", err=True)
            sys.exit(1)

        if status["status"] == "done":
            access_token = status["access_token"]
            host = status.get("api_url", instance_url)

            set_context(host=host, token=access_token)

            click.echo(f"✅ Logged in successfully to {host}")
            click.echo("🔐 Access token stored securely.")
            return


@main.command("context")
@click.argument("selector", required=False)
@click.pass_context
def context_cmd(ctx, selector):
    """Manage CLI contexts.

    Examples:
        scaleout context              # List all contexts
        scaleout context production   # Switch to context named 'production'
        scaleout context 2            # Switch to context at index 2

    """
    contexts = get_all_contexts()
    active = get_active_context()

    # No argument: list all contexts
    if selector is None:
        if not contexts:
            click.echo("No contexts found. Use 'scaleout login' to create a context.")
            return

        click.echo("Available contexts:\n")

        for idx, data in enumerate(contexts):
            name = data.get("name", f"context-{idx}")
            host = data.get("host", "N/A")
            marker = "★" if idx == active else " "
            click.echo(f"  {marker} [{idx}] {name}")
            click.echo(f"      Host: {host}")

        click.echo("\n★ = active context")
        return

    # Argument provided: switch context
    # Check if selector is a number (index-based selection)
    if selector.isdigit():
        idx = int(selector)
        if idx < 0 or idx >= len(contexts):
            click.secho(f"❌ Invalid index {idx}. Available range: 0-{len(contexts) - 1}", fg="red", err=True)
            sys.exit(1)
        target_name = contexts[idx].get("name", f"context-{idx}")
        target_index = idx
    else:
        # Treat as context name
        target_name = selector
        target_index = next((i for i, c in enumerate(contexts) if c.get("name") == target_name), None)

    # Attempt to switch
    if target_index is None:
        click.secho(f"❌ Context '{target_name}' not found.", fg="red", err=True)
        click.echo(f"\nAvailable contexts: {', '.join(c.get('name', f'context-{i}') for i, c in enumerate(contexts))}")
        sys.exit(1)

    if switch_context(target_index):
        context_data = contexts[target_index]
        host = context_data.get("host", "N/A")
        click.echo(f"✅ Switched to context '{target_name}'")
        click.echo(f"   Host: {host}")
    else:
        click.secho(f"❌ Failed to switch to context '{target_name}'", fg="red", err=True)
        sys.exit(1)


@main.command("remove")
@click.argument("selector")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove_cmd(ctx, selector, yes):
    """Remove a CLI context.

    Examples:
        scaleout remove production    # Remove context named 'production'
        scaleout remove 2             # Remove context at index 2
        scaleout remove production -y # Remove without confirmation

    """
    contexts = get_all_contexts()
    active = get_active_context()

    if not contexts:
        click.echo("No contexts found.")
        return

    # Check if selector is a number (index-based selection)
    if selector.isdigit():
        idx = int(selector)
        if idx < 0 or idx >= len(contexts):
            click.secho(f"❌ Invalid index {idx}. Available range: 0-{len(contexts) - 1}", fg="red", err=True)
            sys.exit(1)
        target_name = contexts[idx].get("name", f"context-{idx}")
        target_index = idx
    else:
        # Treat as context name
        target_name = selector
        target_index = next((i for i, c in enumerate(contexts) if c.get("name") == target_name), None)

    # Check if context exists
    if target_index is None:
        click.secho(f"❌ Context '{target_name}' not found.", fg="red", err=True)
        click.echo(f"\nAvailable contexts: {', '.join(c.get('name', f'context-{i}') for i, c in enumerate(contexts))}")
        sys.exit(1)

    # Get context info for display
    context_data = contexts[target_index]
    host = context_data.get("host", "N/A")
    is_active = target_index == active

    # Confirmation prompt (unless --yes flag is used)
    if not yes:
        click.echo(f"\nYou are about to remove context '{target_name}':")
        click.echo(f"  Host: {host}")
        if is_active:
            click.echo("  Status: ⚠️  Currently active")
            if len(contexts) > 1:
                click.echo("  Note: Will automatically switch to another context")
            else:
                click.echo("  Note: This is the last context")

        if not click.confirm("\nAre you sure you want to remove this context?"):
            click.echo("❌ Cancelled.")
            return

    # Attempt to remove
    if remove_context(target_index):
        click.echo(f"✅ Removed context '{target_name}'")

        # Show what happened if it was active
        if is_active:
            new_active = get_active_context()
            if new_active:
                click.echo(f"   Switched to context '{new_active}'")
            else:
                click.echo("   No contexts remaining")
    else:
        click.secho(f"❌ Failed to remove context '{target_name}'", fg="red", err=True)
        sys.exit(1)
