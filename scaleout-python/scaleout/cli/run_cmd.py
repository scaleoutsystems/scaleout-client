import os

import click

from scaleout.cli.main import main

from scaleout.client.importer_package_runtime import ImporterPackageRuntime
from scaleoututil.utils.dispatcher import Dispatcher
from scaleoututil.utils.yaml import read_yaml_file


def check_helper_config_file(config):
    control = config["control"]
    try:
        helper = control["helper"]
    except KeyError:
        click.secho("No helper found in --init settings file.", fg="red")
        exit(-1)
    return helper


def check_yaml_exists(path):
    """Check if scaleout.yaml exists in the given path."""
    yaml_file = os.path.join(path, "scaleout.yaml")
    if not os.path.exists(yaml_file):
        click.secho(f"Could not find scaleout.yaml in {path}", fg="red")
        exit(-1)
    return yaml_file


@main.group(
    "run",
    help="Commands to execute tasks defined in scaleout.yaml.",
    invoke_without_command=True,
)
@click.pass_context
def run_cmd(ctx):
    """Run commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@run_cmd.command("build")
@click.option("-p", "--path", required=True, help="Path to package directory containing scaleout.yaml")
@click.option("-v", "--keep-venv", is_flag=True, required=False, help="Use flag to keep the python virtual environment (python_env in scaleout.yaml)")
@click.option("-d", "--dispatch", is_flag=True, required=False, help="Use flag to indicate that the build should be run in an automatically created venv")
@click.option("-o", "--output", required=False, default=None, help="Output directory for the generated tarball (default: current directory)")
@click.pass_context
def build_cmd(ctx, path, keep_venv, dispatch, output):
    """Execute 'build' entrypoint in scaleout.yaml.

    :param ctx:
    :param path: Path to folder containing scaleout.yaml
    :param keep_venv: Flag to keep the created virtual environment after build
    :param dispatch: Flag to indicate that the build should be run in an automatically created venv
    :param output: Output directory for the generated tarball
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = read_yaml_file(yaml_file)
    # Check that build is defined in scaleout.yaml under entry_points
    if "build" not in config["entry_points"]:
        click.secho("No build command defined in scaleout.yaml", fg="red")
        exit(-1)
    output = output or os.environ.get("SCALEOUT_BUILD_OUTPUT_DIR", os.getcwd())
    output = os.path.abspath(output)
    extra_env = {"SCALEOUT_BUILD_OUTPUT_DIR": output}

    if dispatch:
        dispatcher = Dispatcher(config, path)
        _ = dispatcher.get_or_create_python_env()
        dispatcher.run_cmd("build", extra_env=extra_env)
        if not keep_venv:
            dispatcher.delete_virtual_environment()
    else:
        runtime = ImporterPackageRuntime(None, None)
        runtime.load_local_compute_package(path)
        runtime.run_entrypoint("build")


@run_cmd.command("install")
@click.option("-p", "--path", required=True, help="Path to package directory containing scaleout.yaml")
@click.pass_context
def install_cmd(ctx, path):
    """Install a package environment into active environment.

    :param ctx:
    :param path: Path to folder containing scaleout.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    runtime = ImporterPackageRuntime(None, None)
    runtime.load_local_compute_package(path)
    try:
        runtime.update_runtime_env()
    except Exception as e:
        click.secho(f"An error occurred: {e}", fg="red")
        exit(-1)
    click.secho("Environment installed/updated successfully", fg="green")
