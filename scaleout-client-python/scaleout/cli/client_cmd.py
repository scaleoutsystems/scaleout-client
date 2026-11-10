"""Client commands for the CLI."""

import os
import sys
import uuid
from datetime import datetime

import click
import requests
from scaleoututil.utils.url import build_url, parse_url
import yaml

from scaleout.cli.main import main
from scaleout.cli.shared import apply_config, build_client, call, complement_with_context, render_response
from scaleoututil.logging import ScaleoutLogger
from scaleout.client.connect import ClientOptions
from scaleout.client.dispatcher_client import DispatcherClient
from scaleout.client.importer_client import ImporterClient
from scaleoututil.auth.token_cache import TokenCache
from scaleoututil.auth.login import _enroll_client, _expiry_from_jwt, _decode_jwt_payload

home_dir = os.path.expanduser("~")

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

# --------------- Helper functions --------------- #


def _parse_duration_to_hours(value: str) -> int:
    """Parse a duration string like '24h', '7d', '30m' into whole hours."""
    value = value.strip().lower()
    units = {"h": 1, "d": 24, "m": 1 / 60}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            try:
                amount = float(value[:-1])
                hours = int(amount * multiplier)
                return max(hours, 1)
            except ValueError:
                break
    raise click.BadParameter(f"Invalid duration '{value}'. Use formats like '24h', '7d', or '90m'.")


def _is_enrollment_token(token: str) -> bool:
    """Return True if the token is an enrollment JWT (role=enrollment)."""
    if not token or not isinstance(token, str) or len(token.split(".")) != 3:
        return False
    return _decode_jwt_payload(token).get("role") == "enrollment"


def _is_api_key(token: str) -> bool:
    """Return True if the token is an API key JWT (token_type=api_key)."""
    if not token or not isinstance(token, str) or len(token.split(".")) != 3:
        return False
    return _decode_jwt_payload(token).get("token_type") == "api_key"


def _validate_client_params(config: dict):
    api_url = config["api_url"]
    combiner = config["combiner"]
    combiner_port = config["combiner_port"]
    remote = config.get("package") == "remote"
    if (api_url is None or api_url == "") and (combiner is None or combiner == ""):
        click.echo("Error: Missing required parameter: --api-url or --combiner")
        return False
    if (combiner is not None and combiner != "") and (combiner_port is None or combiner_port == ""):
        click.echo("Error: Missing required parameter: --combiner-port")
        return False
    if remote and (api_url is None or api_url == ""):
        click.echo("Error: Missing required parameter: --api-url for remote package")
        return False
    return True


def _complement_client_params(config: dict) -> None:
    """Ensures that the 'api_url' in the provided configuration dictionary has a protocol (http or https).

    If the 'api_url' does not start with 'http://' or 'https://', it will prepend 'http://' if the URL contains
    'localhost' or '127.0.0.1'. Otherwise, it will prepend 'https://'.
    """
    api_url = config["api_url"]
    scheme, host, port, path = parse_url(api_url)
    port = config.get("api_port") or port

    if scheme is None:
        if host in ["localhost", "127.0.0.1"]:
            scheme = "http"
        else:
            scheme = "https"

        result = build_url(scheme, host, port, path)
        config["api_url"] = result
        click.echo(f"Protocol missing, complementing api_url with protocol: {result}")


def _get_refresh_token_from_enrollment(
    enrollment_token: str, api_url: str, client_id: str | None = None, client_name: str | None = None
) -> tuple[str | None, str | None]:
    """Helper function to enroll a client using an enrollment token and return the resulting refresh token."""
    verify_ssl = not api_url.startswith("http://")
    try:
        enrolled_id, _, refresh_token = _enroll_client(
            api_url,
            enrollment_token,
            client_name=client_name,
            client_id=client_id,
            verify_ssl=verify_ssl,
        )
        click.echo(f"Enrolled as client_id: {enrolled_id}")
        return refresh_token, enrolled_id
    except Exception as e:
        click.echo(f"Enrollment failed: {e}", err=True)
        return None, None


def _get_refresh_token_from_cache(token_cache: TokenCache) -> str:
    """Attempt to load a refresh token from the cache based on the client_id.

    If a valid access token is found in the cache, it will be used along with the cached refresh token.
    If only a refresh token is found, it will be used without an access token.
    If no valid tokens are found, None will be returned.

    Args:
        token_cache (TokenCache): The token cache instance to use.

    Returns:
        str: The refresh token loaded from the cache, or None if not found or invalid.
    """
    try:
        if token_cache.exists():
            cached_data = token_cache.load()
            if not cached_data:
                click.echo("Token cache is empty")
                return None
            return cached_data.get("refresh_token")
    except Exception as e:
        click.echo(f"Warning: Failed to load token cache: {e}")
    return None


# --------------- Commands --------------- #


@main.group(
    "client",
    help="Commands to generate client configs, list or fetch clients, and start a client instance.",
    invoke_without_command=True,
)
@click.pass_context
def client_cmd(ctx):
    """Client commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--path", required=False, help="Path to where client yaml file will be located")
@click.option("--protocol", required=False, default=None, help="Communication protocol of api-server (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of api-server (api)")
@click.option("-P", "--port", required=False, default=None, type=int, help="Port of api-server (api)")
@click.option("-t", "--token", required=False, help="Authentication admin token")
@click.option("-g", "--group", required=False, default=1, help="number of clients to generate in a bulk")
@click.option("-n", "--name", required=True, help="Client name, will be used as prefix for client names. The client name will be suffixed with an index.")
@client_cmd.command("get-config")
@click.pass_context
def create_client(ctx, *, path: str, protocol: str, host: str, port: int, token: str = None, name: str = None, group: int = None):
    """Generate client config file(s).

    The generated client config file(s) contain the following properties:

    - **client_id**: uuid
    - **discover_host**: controller, get from context file
    - **name**: client name (set prefix with options)
    - **refresh_token**: unique refresh token for client
    - **token**: unique access token for client
    """
    discover_host, admin_token = complement_with_context(protocol, host, port, token)

    # Ensure the target directory exists
    try:
        if not path:
            path = os.getcwd()
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            click.echo(f"Path does not exist: {abs_path}")
            click.echo(f"Creating path: {abs_path}")
            os.makedirs(abs_path)
    except PermissionError as e:
        click.echo(f"Error: Permission denied. Details: {e}", fg="red")

    for i in range(group):
        client_id = str(uuid.uuid4())

        # TODO: Fill in with response from token issuer
        response_json = {}

        client_data = {
            "client_id": client_id,
            "discover_host": discover_host,
            "name": f"{name}_{i}",
            "refresh_token": response_json.get("refresh"),
            "token": response_json.get("access"),
        }
        click.echo(f"{i}: Generating client config: {client_data}")
        try:
            client_yaml_path = os.path.join(abs_path, f"{name}_{i}.yaml")
            with open(client_yaml_path, "w") as yaml_file:
                yaml.dump(client_data, yaml_file, default_flow_style=False)
                click.echo(f"{i}: Client config file saved to: {client_yaml_path}")
        except PermissionError as e:
            click.echo(f"Error: Permission denied. Details: {e}", fg="red")
        except Exception as e:
            print(f"Error: Failed to write to YAML file. Details: {e}", fg="red")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("--n_max", required=False, help="Number of items to list")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@client_cmd.command("list")
@click.pass_context
def list_clients(ctx, *, protocol: str, host: str, port: str, token: str = None, n_max: int = None, output_format: str = "human", no_verify_tls: bool = False):
    """List clients.

    **Returns**

    - count: number of clients
    - result: list of clients
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_clients, n_max=n_max)
    return render_response(result, "clients", output_format=output_format, base_url=base_url)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-o", "--output", "output_format", required=False, default="human", help="Output in JSON format")
@click.option("-id", "--id", required=True, help="Client ID")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@client_cmd.command("get")
@click.pass_context
def get_client(ctx, *, protocol: str, host: str, port: str, token: str = None, id: str = None, output_format: str = "human", no_verify_tls: bool = False):
    """Get client.

    **Returns**

    - result: client with given id
    """
    base_url, client = build_client(protocol, host, port, token, no_verify_tls)
    result = call(client.get_client, id)
    return render_response(result, "client", output_format=output_format, base_url=base_url)


@click.option("--enrollment-token", required=True, help="Enrollment JWT issued by an admin.")
@click.option("-u", "--api-url", required=True, help="API server URL (e.g. https://edge.example.com).")
@click.option("-n", "--name", required=False, default=None, help="Optional per-client label stored on the server.")
@click.option(
    "-p",
    "--config-dir",
    required=False,
    default=".",
    help="Directory to write a client YAML config file (named <client_id>.yaml). Defaults to current directory.",
)
@click.option("--client-id", required=False, default=None, help="Optional stable client ID for traceability (e.g. device serial number).")
@click.option("--no-config", is_flag=True, required=False, default=False, help="Disable the generation of config yaml")
@client_cmd.command("enroll")
@click.pass_context
def enroll_client(ctx, *, enrollment_token: str, api_url: str, name: str, config_dir: str, client_id: str, no_config: bool):
    """Enroll a new edge client using an enrollment token.

    Calls the server to register the client, stores the resulting credentials
    in the local token cache, and optionally writes a YAML config file that
    can be passed to ``client start --init``.
    """
    # Infer SSL verification from URL scheme
    verify_ssl = not api_url.startswith("http://")

    # This command's contract is a single clean client_id on stdout (safe for `$(...)`)
    # — see create_enrollment_token above for why logging is redirected rather than left
    # to print to stdout by default when a caller has opted into SCALEOUT_LOG_LEVEL/CONSOLE.
    try:
        with ScaleoutLogger().redirect_to_stderr():
            client_id, access_token, refresh_token = _enroll_client(api_url, enrollment_token, client_name=name, client_id=client_id, verify_ssl=verify_ssl)
    except Exception as e:
        click.echo(f"Enrollment failed: {e}", err=True)
        sys.exit(1)

    # Persist tokens in the local cache keyed by the server-assigned client_id
    cache_dir = os.environ.get("SCALEOUT_TOKEN_CACHE_DIR", None)
    token_cache = TokenCache(cache_id=client_id, cache_dir=cache_dir)
    token_cache.save(access_token, refresh_token, _expiry_from_jwt(access_token))

    click.echo(f"Enrolled successfully. client_id: {client_id}", err=True)

    if config_dir and not no_config:
        try:
            abs_path = os.path.abspath(config_dir)
            os.makedirs(abs_path, exist_ok=True)
            client_name = name or f"client-{client_id[:8]}"
            config_data = {
                "client_id": client_id,
                "discover_host": api_url,
                "name": client_name,
                "refresh_token": refresh_token,
            }
            # Name the config after the stable client_id, not the mutable label.
            yaml_path = os.path.join(abs_path, f"{client_id}.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            click.echo(f"Config written to: {yaml_path}", err=True)
            click.echo(f"Start with: scaleout client start --init {yaml_path}", err=True)
        except Exception as e:
            click.echo(f"Warning: failed to write config file: {e}", err=True)
    else:
        click.echo(f"Start with: scaleout client start --client-id {client_id} --api-url {api_url}", err=True)

    # Print only the client_id to stdout so it can be captured, e.g. CLIENT_ID=$(scaleout client enroll ...)
    click.echo(client_id)


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, type=int, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-n", "--name", required=True, help="Human-readable label for the enrollment token.")
@click.option("--expires-in", default="24h", show_default=True, help="Token lifetime, e.g. '24h', '7d', '90m'.")
@click.option("--no-verify-tls", is_flag=True, default=False, help="Do not verify the server TLS certificate (connection is still encrypted).")
@client_cmd.command("create-enrollment-token")
def create_enrollment_token(*, protocol: str, host: str, port: int, token: str, name: str, expires_in: str, no_verify_tls: bool = False):
    """Create an enrollment token for edge client registration.

    Host and credential resolve the same way as every other command: from
    -H/-t if given, otherwise from the active context ('scaleout login').

    Prints the token to stdout so it can be piped directly to 'scaleout client enroll'.
    """
    expires_in_hours = _parse_duration_to_hours(expires_in)

    # build_client is what every other command uses to resolve -H/-t against the active
    # context; complement_with_context alone only does host/token lookup, not the
    # Login/token-refresh wiring build_client sets up as the client's access_token_provider.
    #
    # This command's contract is a single clean token on stdout (safe for `$(...)`).
    # build_client's own Scaleout(...) construction does a redundant, unused legacy-cache
    # lookup as a side effect and logs it via ScaleoutLogger, which — whenever a caller has
    # opted into SCALEOUT_LOG_LEVEL/SCALEOUT_LOG_CONSOLE — writes to stdout by default and
    # would corrupt the captured token; redirect it to stderr instead for this call.
    try:
        with ScaleoutLogger().redirect_to_stderr():
            base_url, client = build_client(protocol, host, port, token, no_verify_tls)
            headers = client._get_headers()
    except Exception as e:
        click.echo(f"Error: Failed to resolve host/credentials: {e}", err=True)
        sys.exit(1)

    if "Authorization" not in headers:
        click.echo("Error: No token found. Pass -t/--token or set an active context with 'scaleout login'.", err=True)
        sys.exit(1)

    url = base_url.rstrip("/") + "/api/v1/auth/enrollment-tokens"
    try:
        click.echo(f"Creating enrollment token '{name}' (expires in {expires_in_hours}h)...", err=True)
        resp = requests.post(
            url,
            json={"name": name, "expires_in_hours": expires_in_hours},
            headers=headers,
            timeout=10,
            verify=client.verify,
        )
        if resp.status_code == 401:
            click.echo("Error: Unauthorized. Check your token or log in again.", err=True)
            sys.exit(1)
        resp.raise_for_status()
        data = resp.json()
        enrollment_token = data.get("token")
        expires_at = data.get("expires_at", "")
        click.echo(f"Enrollment token created (name={name}, expires={expires_at})", err=True)
        # Print only the token to stdout for clean piping
        click.echo(enrollment_token)
    except requests.RequestException as e:
        click.echo(f"Error: Failed to create enrollment token: {e}", err=True)
        sys.exit(1)


@client_cmd.command("start")
@click.option("-u", "--api-url", required=False, help="Hostname for scaleout api.")
@click.option("-p", "--api-port", required=False, help="Port for discovery services (reducer).")
@click.option(
    "--token",
    required=False,
    help=(
        "Authentication token: a client refresh token (exchanged automatically for an access token), or an enrollment "
        "token to self-enroll on the fly. Prefer 'scaleout client enroll' plus --client-id/--init for anything that "
        "needs a persistent identity across restarts; passing an enrollment token here is best kept for ephemeral/CI nodes."
    ),
)
@click.option("-n", "--name", required=False)
@click.option("-i", "--client-id", required=False)
@click.option(
    "--remote-package",
    is_flag=False,
    flag_value="__default__",
    default=None,
    help=(
        "Download and extract compute package from server (managed python env is enabled by default). "
        "Optionally pass a package name to fetch a specific package, e.g. --remote-package my-package."
    ),
)
@click.option(
    "--log-level",
    required=False,
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option("-c", "--preferred-combiner", type=str, required=False, default="", help="name of the preferred combiner")
@click.option("--combiner", type=str, required=False, default=None, help="Skip combiner assignment from discover service and attach directly to combiner host.")
@click.option("--combiner-port", type=str, required=False, default=None, help="Combiner port, need to be used with --combiner")
@click.option("-va", "--validator", required=False, default=None)
@click.option("-tr", "--trainer", required=False, default=None)
@click.option("-hp", "--helper_type", required=False, default=None)
@click.option("-in", "--init", required=False, default=None, help="Set to a filename to (re)init client from file state.")
@click.option("--dispatcher", is_flag=True, help="Use the dispatcher client instead of the importer client.")
@click.option("--disable-managed-env", is_flag=True, help="Disable managed python environment")
@click.pass_context
def client_start_cmd(
    ctx,
    *,
    api_url: str,
    api_port: int,
    token: str,
    name: str,
    client_id: str,
    remote_package: str,
    log_level: str,
    preferred_combiner: str,
    combiner: str,
    combiner_port: int,
    validator: bool,
    trainer: bool,
    helper_type: str,
    init: str,
    dispatcher: bool,
    disable_managed_env: bool = False,
):
    """Start client.

    For a persistent client identity, enroll first with ``scaleout client enroll`` and start via
    ``--client-id``/``--init`` (see the CLI docs). Passing an enrollment token directly via --token
    self-enrolls a fresh, ephemeral client identity on every invocation.
    """
    use_remote_package = remote_package is not None
    package_name = remote_package if use_remote_package and remote_package != "__default__" else None
    package = "remote" if use_remote_package else "local"
    managed_env = not disable_managed_env

    if package_name:
        click.echo(f"Using remote compute package: {package_name}")

    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        click.echo(f"Invalid log level: {log_level}. Defaulting to INFO.")
        log_level = "INFO"

    ScaleoutLogger().set_log_level_from_string(log_level)

    config = {
        "api_url": None,
        "api_port": None,
        "refresh_token": None,
        "name": None,
        "client_id": None,
        "preferred_combiner": None,
        "combiner": None,
        "combiner_port": None,
        "validator": None,
        "trainer": None,
        "package_checksum": None,
        "helper_type": None,
        # to cater for old inputs
        "discover_host": None,
        "discover_port": None,
    }

    if init:
        apply_config(init, config)
        click.echo(f"Client configuration loaded from file: {init}")

        # to cater for old inputs
        if config["discover_host"] is not None:
            config["api_url"] = config["discover_host"]

        if config["discover_port"] is not None:
            config["api_port"] = config["discover_port"]

    # Override client_id if provided via CLI, otherwise rely on config file or generated value. This is needed early for token cache lookup.
    # If not client id is provided, we'll generate a random one.
    if client_id and client_id.strip():
        config["client_id"] = client_id
    elif config["client_id"] is None:
        config["client_id"] = str(uuid.uuid4())

    if name and name != "":
        config["name"] = name
        if config["name"]:
            click.echo(f"Input param name: {name} overrides value from file")
    elif config["name"] is None:
        config["name"] = f"client-{config['client_id'][:8]}"

    if api_url and api_url.strip():
        config["api_url"] = api_url

    _complement_client_params(config)

    # Fail early and clearly if the API URL still has no scheme. Without this,
    # a scheme-less URL only surfaces much later as an opaque requests error
    # ("Invalid URL '<host>/api/v1/...': No scheme supplied") during token
    # refresh. _complement_client_params should normally prevent this, so
    # reaching here means the value could not be normalized.
    if config["api_url"] and not str(config["api_url"]).startswith(("http://", "https://")):
        click.echo(
            f"Error: --api-url '{config['api_url']}' is missing a scheme. Use a full URL, e.g. http://{config['api_url']} or https://{config['api_url']}."
        )
        return

    token_cache: TokenCache | None = None

    if os.environ.get("SCALEOUT_PERSIST_TOKENS", "true").lower() not in ("false", "0", "no"):
        token_cache = TokenCache(cache_id=config["client_id"], cache_dir=os.environ.get("SCALEOUT_TOKEN_CACHE_DIR", None))

    # Getting the refresh token.
    # If an enrollment token is provided via CLI, it takes precedence and will be exchanged for client credentials before starting the client.
    # If a regular token is provided via CLI, it will be used directly.
    # If no token is provided via CLI but a client_id is available (either from CLI or config file),
    # the token cache will be checked for a refresh token associated with that client_id.

    refresh_token: str | None = None
    if _is_api_key(token):
        # API keys are user-scoped credentials meant for the SDK / CI use (Scaleout(token=...)).
        # They are a poor fit for edge clients: they have no enrolled-client record, so they
        # cannot be revoked per-client (revoking the key kills every client sharing it), and
        # each REST call incurs a database lookup. Edge clients should enroll instead.
        raise click.ClickException(
            "An API key cannot be used to start an edge client. API keys are for SDK/CI use and "
            "cannot be revoked per-client.\nEnroll the client instead:\n"
            "  scaleout client create-enrollment-token --name <label>\n"
            "  scaleout client enroll --enrollment-token <enrollment-token> -u <api-url>\n"
            "  scaleout client start --client-id <client-id> -u <api-url>"
        )
    if _is_enrollment_token(token):
        refresh_token, enrolled_id = _get_refresh_token_from_enrollment(
            token, api_url=config["api_url"], client_id=config["client_id"], client_name=config["name"]
        )
        config["client_id"] = (
            enrolled_id  # Override client_id with enrolled_id if enrollment is successful. This should only happen if no client_id was provided.
        )

    if not refresh_token and token and token.strip():
        refresh_token = token

    if not refresh_token and token_cache:
        refresh_token = _get_refresh_token_from_cache(token_cache=token_cache)

    config["refresh_token"] = refresh_token

    # NOTE: api_url is intentionally not re-applied here. It is set from the CLI
    # at the top of this command (before _complement_client_params), so the
    # protocol-complemented value (e.g. "http://localhost") is authoritative.
    # Re-assigning the raw CLI value here would clobber that scheme and make
    # the client connect to a scheme-less URL, failing token refresh.

    if api_port:
        config["api_port"] = api_port
        if config["api_port"]:
            click.echo(f"Input param api_port: {api_port} overrides value from file")

    if preferred_combiner and preferred_combiner != "":
        config["preferred_combiner"] = preferred_combiner
        if config["preferred_combiner"]:
            click.echo(f"Input param preferred_combiner: {preferred_combiner} overrides value from file")

    if combiner and combiner != "":
        config["combiner"] = combiner
        if config["combiner"]:
            click.echo(f"Input param combiner: {combiner} overrides value from file")

    if combiner_port:
        config["combiner_port"] = combiner_port
        if config["combiner_port"]:
            click.echo(f"Input param combiner_port: {combiner_port} overrides value from file")

    if validator is not None:
        config["validator"] = validator
        if config["validator"] is not None:
            click.echo(f"Input param validator: {validator} overrides value from file")
    elif config["validator"] is None:
        config["validator"] = True

    if trainer is not None:
        config["trainer"] = trainer
        if config["trainer"] is not None:
            click.echo(f"Input param trainer: {trainer} overrides value from file")
    elif config["trainer"] is None:
        config["trainer"] = True

    if helper_type and helper_type != "":
        config["helper_type"] = helper_type
        if config["helper_type"]:
            click.echo(f"Input param helper_type: {helper_type} overrides value from file")

    if not _validate_client_params(config):
        return

    client_options = ClientOptions(
        name=config["name"],
        package=package,
        preferred_combiner=config["preferred_combiner"],
        client_id=config["client_id"],
    )

    # Create token update callback to save tokens to cache
    def on_token_refresh(access_token: str, refresh_token: str, expires_at: datetime) -> None:
        """Callback to save tokens when they are refreshed."""
        if token_cache:
            try:
                token_cache.save(access_token, refresh_token, expires_at)
                ScaleoutLogger().debug(f"Tokens updated in cache: {token_cache.cache_file}")
                click.echo(f"Tokens updated in cache: {token_cache.cache_file}")
            except Exception as e:
                click.echo(f"Warning: Failed to save tokens to cache: {e}")

    if dispatcher:
        client = DispatcherClient(
            api_url=config["api_url"],
            client_obj=client_options,
            combiner_host=config["combiner"],
            combiner_port=config["combiner_port"],
            access_token=config.get("access_token"),
            refresh_token=config["refresh_token"],
            package_checksum=config["package_checksum"],
            package_name=package_name,
            helper_type=config["helper_type"],
            token_refresh_callback=on_token_refresh,
        )
    else:
        client = ImporterClient(
            api_url=config["api_url"],
            client_obj=client_options,
            combiner_host=config["combiner"],
            combiner_port=config["combiner_port"],
            access_token=config.get("access_token"),
            refresh_token=config["refresh_token"],
            package_checksum=config["package_checksum"],
            package_name=package_name,
            helper_type=config["helper_type"],
            managed_env=managed_env,
            token_refresh_callback=on_token_refresh,
        )

    client.start()
