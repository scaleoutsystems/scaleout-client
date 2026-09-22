Scaleout Edge Client: Dockerized Example (with CUDA support)
--------------------------------------------------------------

This example packages the Scaleout Edge Python client as a Docker
container, so you can run an edge client without installing Python
dependencies on the host, and enable GPU training via the NVIDIA
Container Toolkit.

It bundles the same model as the ``mnist-pytorch`` example. Use it as a
template: replace the contents of ``client/`` with your own
``model.py``, ``data.py`` and ``startup.py``, and point ``data/`` at your
own training data.

For the full walkthrough (including GPU/CUDA host setup), see
`Client Docker & CUDA <https://docs.scaleoutsystems.com/en/latest/client-docker.html>`__.

.. important::

   ``scaleout`` and ``scaleoututil`` are built from the source in this
   repository, not installed from PyPI — this lets you run a client from a
   branch that hasn't been released yet. That means the Docker build needs
   ``scaleout-util/`` and ``scaleout-client-python/`` present as siblings at
   the repo root: **clone the full repository**, don't copy just this
   ``docker-client`` folder out on its own.

Prerequisites
-------------

- A full clone of this repository (see above — not just this folder).
- `Docker Engine 24+ <https://docs.docker.com/engine/install/>`__ with the Compose plugin.
- For GPU training: an NVIDIA GPU, a recent NVIDIA driver, and the
  `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__
  installed on the host.

Quickstart (CPU)
----------------

1. Copy the environment template and fill in your instance URL, a stable
   client id, and a short-lived enrollment token (create one with
   ``scaleout client create-enrollment-token --name docker-example``):

   .. code-block::

       cp .env.example .env
       # edit .env

2. Build and start the client (run from this directory):

   .. code-block::

       docker compose up --build

   On the first run the client enrolls using ``SCALEOUT_TOKEN`` and caches
   a refresh token in the ``scaleout_tokens`` volume. After a successful
   first run, remove ``SCALEOUT_TOKEN`` from ``.env`` — restarts reuse the
   cached refresh token automatically.

Running with GPU/CUDA
----------------------

Once the NVIDIA Container Toolkit is installed and verified on the host
(see the docs page linked above), start with the GPU overlay instead:

.. code-block::

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

No changes to the image are required — the container has no CUDA toolkit
baked in. GPU support comes entirely from the host exposing the device to
the container; ``torch`` (installed dynamically from
``client/python_env.yaml`` into a managed venv when the client starts)
already ships its own CUDA runtime.

Adapting this to your own ML code
----------------------------------

Replace the files under ``client/`` with your own project, following the
same layout as the other ``examples/*/client`` directories in this repo:

.. code-block::

    client/
    |-- scaleout.yaml       # entry points: build / startup
    |-- python_env.yaml     # your ML framework + dependencies
    |-- startup.py          # wires train/validate callbacks
    |-- model.py            # your model definition
    `-- data.py             # your data loading

Two things specific to this Docker setup, not present in the plain
(non-Docker) examples:

- ``client/`` is bind-mounted **read-only** (``docker-compose.yml``), since
  it's your versioned code. Anything your code needs to *write* (downloaded
  or preprocessed data, caches) must go somewhere else — this example's
  ``data.py`` reads the ``SCALEOUT_DATA_DIR`` environment variable (set to
  ``/app/data`` in ``docker-compose.yml``) instead of writing next to
  itself. Follow the same pattern in your own ``data.py``.
- Training data lives under ``./data`` on the host, bind-mounted to
  ``/app/data``. Point your own client at your own dataset by changing what
  you mount there — no image rebuild needed.

Environment variables
----------------------

See ``.env.example`` for the full list. In short: ``SCALEOUT_API_URL`` and
``SCALEOUT_CLIENT_ID`` are always required; ``SCALEOUT_TOKEN`` (an
enrollment token) is only needed for the first run.
