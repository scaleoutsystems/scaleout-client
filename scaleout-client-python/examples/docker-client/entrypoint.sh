#!/bin/sh
# Builds and runs `scaleout client start` from environment variables.
# See .env.example for what each one does.
set -eu

: "${SCALEOUT_API_URL:?SCALEOUT_API_URL must be set (see .env.example)}"
: "${SCALEOUT_CLIENT_ID:?SCALEOUT_CLIENT_ID must be set (see .env.example)}"

# On an aarch64 host with the GPU actually passed through (e.g. NVIDIA Jetson),
# PyPI has no CUDA-enabled torch wheel: `pip install torch` silently resolves
# to a CPU-only build there. Point pip at NVIDIA's own index in that case only
# (skipped on x86_64, and on aarch64 without a GPU) so python_env.yaml's plain
# `pip install torch` picks up a CUDA build instead. See docs/client-docker.rst.
if [ "$(uname -m)" = "aarch64" ] && command -v nvidia-smi >/dev/null 2>&1; then
    export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu132}"
    export PIP_PRE="${PIP_PRE:-1}"
fi

set -- scaleout client start --api-url "$SCALEOUT_API_URL" --client-id "$SCALEOUT_CLIENT_ID"

if [ -n "${SCALEOUT_TOKEN:-}" ]; then
    set -- "$@" --token "$SCALEOUT_TOKEN"
fi

if [ -n "${SCALEOUT_CLIENT_NAME:-}" ]; then
    set -- "$@" --name "$SCALEOUT_CLIENT_NAME"
fi

exec "$@"
