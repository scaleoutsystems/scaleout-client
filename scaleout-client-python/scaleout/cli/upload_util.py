import os
import click
import requests
import sys

from scaleout.cli.shared import get_api_url


def perform_chunked_upload(base_url: str, token: str, file_path: str, headers: dict) -> str:
    """Uploads a file in chunks using the Scaleout chunked upload endpoints and returns a file_token.

    Args:
        base_url: The controller's API base URL.
        token: The active user's authentication token.
        file_path: Absolute or relative path to the file on disk.
        headers: Base HTTP headers (e.g. Authorization) to inject into every request.

    Returns:
        The `file_token` string that resolves to this file inside the backend registry.
    """
    if not os.path.exists(file_path):
        click.secho(f"Upload failed: File not found ({file_path})", fg="red")
        sys.exit(1)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Initial chunk size: 900 KB (fits in nginx 1 MB default)
    current_chunk_size = 900 * 1024
    min_chunk_size = 256 * 1024

    while True:
        try:
            return _do_chunked_upload(base_url, token, file_path, headers, file_name, file_size, current_chunk_size)
        except requests.exceptions.HTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 413:
                next_size = current_chunk_size // 2
                if next_size < min_chunk_size:
                    click.secho(f"\nUpload failed: proxy rejects all supported chunk sizes (minimum {min_chunk_size // 1024} KB).", fg="red")
                    sys.exit(1)
                click.secho(f"\nProxy rejected chunk size (413). Retrying with {next_size // 1024} KB chunks...", fg="yellow")
                current_chunk_size = next_size
                continue
            # Non-413 HTTP errors: report and exit
            click.secho(f"Upload failed: {e}", fg="red")
            if getattr(e, "response", None) is not None:
                click.secho(f"Details: {e.response.text}", fg="red")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            # Other request errors: report and exit
            click.secho(f"Upload failed: {e}", fg="red")
            sys.exit(1)


def _do_chunked_upload(base_url: str, token: str, file_path: str, headers: dict, file_name: str, file_size: int, chunk_size: int) -> str:
    click.secho(f"Initializing chunked upload for {file_name} ({file_size} bytes, {chunk_size // 1024} KB chunks)", fg="cyan")

    # 1. Initialize Upload
    init_url = get_api_url(base_url, "file-upload/init")
    try:
        init_response = requests.post(init_url, json={"file_name": file_name, "file_size": file_size, "chunk_size": chunk_size}, headers=headers, verify=False)
        init_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        click.secho(f"Failed to initialize chunked upload: {e}", fg="red")
        if getattr(e, "response", None) is not None:
            click.secho(f"Details: {e.response.text}", fg="red")
        sys.exit(1)

    upload_id = init_response.json().get("upload_id")
    if not upload_id:
        click.secho("Failed to receive an upload_id from the backend", fg="red")
        sys.exit(1)

    # 2. Upload Chunks
    chunk_url = get_api_url(base_url, f"file-upload/{upload_id}/chunk")

    with click.progressbar(length=file_size, label="Uploading") as bar, open(file_path, "rb") as f:
        chunk_index = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break

            chunk_headers = headers.copy()
            chunk_headers["X-Chunk-Index"] = str(chunk_index)

            # Do not set standard headers strictly since requests calculates content-length for binary itself.
            try:
                chunk_resp = requests.post(chunk_url, data=chunk_data, headers=chunk_headers, verify=False)
                chunk_resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # If 413, bubble up to trigger resize
                if e.response.status_code == 413:
                    # Best effort: abort the session
                    abort_url = get_api_url(base_url, f"file-upload/{upload_id}/abort")
                    requests.post(abort_url, headers=headers, verify=False)
                    raise e
                raise e
            except requests.exceptions.RequestException as e:
                click.secho(f"\nFailed to upload chunk {chunk_index}: {e}", fg="red")
                if getattr(e, "response", None) is not None:
                    click.secho(f"Details: {e.response.text}", fg="red")
                sys.exit(1)

            bar.update(len(chunk_data))
            chunk_index += 1

    # 3. Complete Upload
    click.secho("\nFinalizing upload...", fg="cyan")
    complete_url = get_api_url(base_url, f"file-upload/{upload_id}/complete")

    try:
        complete_resp = requests.post(complete_url, headers=headers, verify=False)
        complete_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        click.secho(f"Failed to finalize chunked upload: {e}", fg="red")
        if getattr(e, "response", None) is not None:
            click.secho(f"Details: {e.response.text}", fg="red")
        sys.exit(1)

    json_resp = complete_resp.json()
    file_token = json_resp.get("file_token")
    if not file_token:
        click.secho("Failed to fetch file_token. Backend completed upload but emitted no token.", fg="red")
        sys.exit(1)

    return file_token
