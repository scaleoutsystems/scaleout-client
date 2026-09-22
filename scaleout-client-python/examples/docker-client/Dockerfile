# Builds the Scaleout Edge Python client as a container image, installing
# `scaleout`/`scaleoututil` from the source checked out alongside this
# example, not from PyPI. This lets you build and run a client from a dev
# branch before it's released -- the tradeoff is that the build context is
# the repo root (see docker-compose.yml's `build.context`), not this
# directory alone: `scaleout-client-python/` and `scaleout-util/` must be
# present as siblings, i.e. you need a full clone of this repo, not just
# this example folder.
#
# No ML framework or CUDA toolkit is installed here: `scaleout client start`
# creates a managed virtualenv and installs whatever `client/python_env.yaml`
# declares (e.g. torch) at *runtime*, once a client/ directory is mounted in.
# GPU support therefore comes entirely from the host + container runtime
# (NVIDIA Container Toolkit) exposing the GPU, not from anything baked in
# here. See ../../../docs/client-docker.rst.
#
# Build via docker compose (from this directory) rather than a bare
# `docker build`, since the build context must be the repo root:
#   docker compose up --build

ARG BASE_IMG=python:3.12-slim
FROM $BASE_IMG

# torch's Linux wheels dlopen libgomp.so.1 at import time, which the slim
# base image does not include by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY scaleout-util /tmp/scaleout-util
COPY scaleout-client-python /tmp/scaleout-client-python
RUN pip install --no-cache-dir /tmp/scaleout-util /tmp/scaleout-client-python \
    && rm -rf /tmp/scaleout-util /tmp/scaleout-client-python

WORKDIR /app
COPY scaleout-client-python/examples/docker-client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
