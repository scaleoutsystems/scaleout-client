import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import click
from scaleoututil.utils.url import build_url, assemble_endpoint_url, parse_url
import requests
import yaml
import json

from scaleoututil.logging import ScaleoutLogger

CONTROLLER_DEFAULTS = {"local_protocol": "http", "protocol": "https", "host": "localhost", "port": 8092, "debug": False}


def _get_validated_home_dir() -> str:
    """Get and validate the home directory for Scaleout configuration.

    Returns:
        str: Absolute path to the validated home directory.

    Raises:
        RuntimeError: If the home directory cannot be determined or validated.

    """
    # Get the home directory from environment or use user's home
    home_dir_str = os.environ.get("SCALEOUT_HOME_DIR", os.path.expanduser("~"))

    if not home_dir_str:
        raise RuntimeError("Unable to determine home directory. Please set SCALEOUT_HOME_DIR environment variable.")

    try:
        # Convert to Path and resolve to absolute path
        home_path = Path(home_dir_str).resolve()

        # Validate the path
        if home_path.exists():
            # Check if it's a directory
            if not home_path.is_dir():
                raise RuntimeError(f"Home directory path exists but is not a directory: {home_path}")
            # Check if writable
            if not os.access(home_path, os.W_OK):
                raise RuntimeError(f"Home directory is not writable: {home_path}")
        else:
            # Try to create the directory if it doesn't exist
            try:
                home_path.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"Cannot create home directory {home_path}: {e}")

        return str(home_path)
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Invalid home directory path '{home_dir_str}': {e}")


HOME_DIR = _get_validated_home_dir()

API_VERSION = "v1"

CONTEXT_FOLDER = ".scaleout"


def apply_config(path: str, config: dict):
    """Parse client config from file.

    Override configs from the CLI with settings in config file.

    :param config: Client config (dict).
    """
    with open(path, "r") as file:
        try:
            settings = dict(yaml.safe_load(file))
        except Exception:
            click.secho("Failed to read config from settings file, exiting", fg="red")
            return

    for key, val in settings.items():
        config[key] = val


def get_api_url(base_url: str, endpoint: str, query: dict = None) -> str:
    """Utility function to build api url based on provided information."""
    return assemble_endpoint_url(base_url, f"api/{API_VERSION}", endpoint, **(query or {}))


# Print response from api (list of entities)
def process_response(response, entity_name: str, output_format: str = "human", base_url: Optional[str] = None):
    """Prints the api response to the cli.
    :param response:
        type: array
        description: list of entities
    :param entity_name:
        type: string
        description: name of entity
    :param output_format:
        type: string
        description: output format (human, json, yaml)
    :param base_url:
        type: string
        description: base URL of the API server
    :return:
        type: int
        description: return code (0=success, 1=failure)
    """
    if response is None:
        if base_url:
            click.secho(f"Error: Could not connect to the controller API at {base_url}", fg="red")
        else:
            click.secho("Error: Could not connect to the controller API", fg="red")
        return 1
    if response.status_code == 200:
        json_data = response.json()
        if json_data.get("count") is not None and json_data.get("result") is not None:
            if output_format == "json":
                click.echo(json.dumps(json_data["result"]))
                return 0
            elif output_format == "yaml":
                click.echo(yaml.dump(json_data["result"]))
                return 0
            elif output_format != "human":
                click.secho(f"Error: Unsupported output format: {output_format}", fg="red")
                return 1
            # list output
            count = json_data.get("count")
            result = json_data.get("result")
            click.echo(f"Found {count} {entity_name} at {base_url}")
            click.echo("\n---------------------------------\n")
            for obj in result:
                click.echo("{")
                for k, v in obj.items():
                    click.echo(f"\t{k}: {v}")
                click.echo("}")
        else:
            if output_format == "json":
                click.echo(json.dumps(json_data))
                return 0
            elif output_format == "yaml":
                click.echo(yaml.dump(json_data))
                return 0
            elif output_format != "human":
                click.secho(f"Error: Unsupported output format: {output_format}", fg="red")
                return 1
            click.echo(f"Found {entity_name} at {base_url}")
            click.echo("\n---------------------------------\n")
            for k, v in json_data.items():
                click.echo(f"\t{k}: {v}")
    elif response.status_code == 500:
        json_data = response.json()
        click.secho(f"Error: {json_data['message']}" + (f" at {base_url}" if base_url else ""), fg="red")
        return 1
    else:
        click.secho(f"Error: {response.status_code}" + (f" at {base_url}" if base_url else ""), fg="red")
        return 1
    return 0


def set_context(host, token, name: Optional[str] = None):
    """Saves context data and sets it as active"""
    # If no name provided, use host as the name
    if not name:
        # Extract a readable name from the host URL
        try:
            parsed = urlparse(host)
            name = parsed.netloc or host
        except Exception:
            name = host

    context_data = {
        "name": name,
        "host": host,
        "token": token,
    }

    try:
        if not os.path.exists(f"{HOME_DIR}/{CONTEXT_FOLDER}"):
            os.makedirs(f"{HOME_DIR}/{CONTEXT_FOLDER}")

        # Load existing contexts
        contexts_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/contexts.yaml"
        if os.path.exists(contexts_file):
            with open(contexts_file, "r") as f:
                all_contexts = yaml.safe_load(f) or []
                # Handle migration from old dict format to new list format
                if isinstance(all_contexts, dict):
                    all_contexts = []
        else:
            all_contexts = []

        # Check if context with this name AND host already exists
        target_index = None
        for i, context in enumerate(all_contexts):
            if context.get("name") == name and context.get("host") == host:
                target_index = i
                break

        if target_index is not None:
            # Update existing context
            all_contexts[target_index] = context_data
            new_index = target_index
        else:
            # Add new context to the list
            all_contexts.append(context_data)
            new_index = len(all_contexts) - 1

        # Save all contexts
        with open(contexts_file, "w") as yaml_file:
            yaml.dump(all_contexts, yaml_file, default_flow_style=False)

        # Set as active context by index
        set_active_context(new_index)

    except Exception as e:
        click.secho(f"Error: Failed to write to YAML file. Details: {e}")


def set_active_context(index: int):
    """Sets the active context by index"""
    active_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/active.yaml"
    try:
        # Ensure the context directory exists before writing the active context file
        context_dir = f"{HOME_DIR}/{CONTEXT_FOLDER}"
        if not os.path.exists(context_dir):
            os.makedirs(context_dir, exist_ok=True)
        with open(active_file, "w") as f:
            yaml.dump({"active_index": index}, f)
    except Exception as e:
        click.secho(f"Error: Failed to set active context. Details: {e}")


def get_active_context() -> Optional[int]:
    """Gets the index of the active context"""
    active_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/active.yaml"
    try:
        if os.path.exists(active_file):
            with open(active_file, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    # Support both old 'active' (name) and new 'active_index' (index) formats
                    if "active_index" in data:
                        return data.get("active_index")
                    # Migration: if old format exists, return None to trigger fallback
                    return None
    except Exception:
        pass
    return None


def get_all_contexts() -> list:
    """Retrieves all saved contexts as a list"""
    contexts_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/contexts.yaml"
    try:
        if os.path.exists(contexts_file):
            with open(contexts_file, "r") as f:
                contexts = yaml.safe_load(f) or []
                # Handle migration from old dict format to new list format
                if isinstance(contexts, dict):
                    # Convert old dict format to list format
                    return [{"name": name, **data} for name, data in contexts.items()]
                return contexts
    except Exception as e:
        click.secho(f"Error: Failed to read contexts file. Details: {e}")
    return []


def get_context_by_index(index: int) -> Optional[dict]:
    """Retrieves a specific context by index"""
    contexts = get_all_contexts()
    if not contexts or index < 0 or index >= len(contexts):
        return None
    return contexts[index]


def switch_context(index: int) -> bool:
    """Switches to a different context by index"""
    context_data = get_context_by_index(index)
    if not context_data:
        return False

    try:
        # Update legacy context.yaml for backward compatibility
        with open(f"{HOME_DIR}/{CONTEXT_FOLDER}/context.yaml", "w") as yaml_file:
            yaml.dump(context_data, yaml_file, default_flow_style=False)

        # Set as active
        set_active_context(index)
        return True
    except Exception as e:
        click.secho(f"Error: Failed to switch context. Details: {e}")
        return False


def remove_context(index: int) -> bool:
    """Removes a context by index"""
    contexts_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/contexts.yaml"

    try:
        # Load existing contexts
        if not os.path.exists(contexts_file):
            return False

        all_contexts = get_all_contexts()

        # Check if context exists
        if index < 0 or index >= len(all_contexts):
            return False

        # Check if it's the active context
        active_index = get_active_context()
        removed_context_name = all_contexts[index].get("name", f"context-{index}")

        # Remove the context
        all_contexts.pop(index)

        # Save updated contexts
        with open(contexts_file, "w") as f:
            yaml.dump(all_contexts, f, default_flow_style=False)

        # If we removed the active context, clear the active context or switch to another
        if active_index == index:
            if all_contexts:
                # Switch to the first available context (index 0)
                if not switch_context(0):
                    click.secho(
                        f"Error: Removed context '{removed_context_name}', but failed to switch to context at index 0.",
                        fg="red",
                    )
                    return False
            else:
                # No contexts left, clear active
                active_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/active.yaml"
                if os.path.exists(active_file):
                    os.remove(active_file)
                # Clear legacy context file
                legacy_file = f"{HOME_DIR}/{CONTEXT_FOLDER}/context.yaml"
                if os.path.exists(legacy_file):
                    os.remove(legacy_file)
        elif active_index is not None and active_index > index:
            # Adjust active index if it's after the removed context
            set_active_context(active_index - 1)

        return True
    except Exception:
        # Let the caller handle user-facing error messages based on the False return value.
        return False


def get_context(context_path: str) -> dict:
    """Retrieves context data from yaml file in given path"""
    try:
        with open(context_path, "r") as yaml_file:
            context_data = yaml.safe_load(yaml_file)
    except Exception as e:
        click.secho(f"Error: Failed to read YAML file. Details: {e}")
        context_data = {}
    return context_data


def get_response(base_url: str, endpoint: str, query: dict, token: str, headers: dict):
    """Utility function to retrieve response from get request based on provided information."""
    url = get_api_url(base_url=base_url, endpoint=endpoint, query=query)
    _token = get_scheme_token(token)
    if _token:
        headers["Authorization"] = _token
    try:
        response = requests.get(url, headers=headers)
    except requests.exceptions.RequestException:
        response = None
    return response


def patch_request(base_url: str, endpoint: str, query: dict, token: str, data: dict, headers: dict):
    """Utility function to retrieve response from patch request based on provided information."""
    url = get_api_url(base_url=base_url, endpoint=endpoint, query=query)
    _token = get_scheme_token(token)
    if _token:
        headers["Authorization"] = _token

    response = requests.patch(url, json=data, headers=headers)

    return response


def complement_with_context(protocol: Optional[str], host: Optional[str], port: Optional[str], token: Optional[str]) -> Tuple[str, Optional[str]]:
    """Utility function to retrieve host from contexts file."""
    if not host:
        if protocol or port:
            raise ValueError("Both protocol and port must be provided together with host.")

        # Get the active context index
        active_index = get_active_context()
        context_data = {}

        if active_index is not None:
            # Get the specific context by index
            context_data = get_context_by_index(active_index) or {}
            host = context_data.get("host")
            if host:
                context_name = context_data.get("name", f"context-{active_index}")
                ScaleoutLogger().debug(f"Using host: '{host}' from context '{context_name}' (index {active_index})")
        else:
            # No active context, try to get the first context
            all_contexts = get_all_contexts()
            if all_contexts:
                # Use the first available context (index 0)
                context_data = all_contexts[0]
                host = context_data.get("host")
                if host:
                    context_name = context_data.get("name", "context-0")
                    ScaleoutLogger().debug(f"Using host: '{host}' from context '{context_name}' (index 0)")
            else:
                host = None

        # TODO: which prioritizes env vs context file?
        token = token or context_data.get("token") or os.environ.get("SCALEOUT_AUTH_TOKEN")

    host = complement_with_defaults(protocol, host, port)
    return host, token


def complement_with_defaults(protocol: Optional[str], host: Optional[str], port: Optional[str]) -> str:
    """Utility function to build host URL with defaults when necessary."""
    if host:
        _protocol, _host, _port, _ = parse_url(host)
        protocol = protocol or _protocol
        port = port or _port
        if _host in ["localhost", "127.0.0.1"]:
            if protocol is None:
                protocol = CONTROLLER_DEFAULTS["local_protocol"]
        elif protocol is None:
            protocol = CONTROLLER_DEFAULTS["protocol"]
        host = _host
        host = build_url(protocol, host, port, "")
    return host


def get_scheme_token(token: str):
    if token:
        token_scheme = os.environ.get("SCALEOUT_AUTH_SCHEME", "Bearer")
        token = f"{token_scheme} {token}"
    return token
