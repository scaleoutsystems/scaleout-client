Scaleout Edge Project: CORe50 New-Domains (PyTorch, Continual Learning)
-----------------------------------------------------------------------

This example demonstrates **federated continual learning** on the CORe50 mini
benchmark in its "new instances" (a.k.a. new-domains) scenario, downloaded via
`Avalanche <https://avalanche.continualai.org/>`_:

* **Eight training domains.** Each federated round advances every participating
  client to one new domain, with no replay. The domain is derived from the
  server's round counter as ``(round_id - 1) % 8``. That counter is global to
  the deployment, not per-session, so only the first session on a fresh
  deployment starts at domain 0; a later 8-round session still covers all eight
  domains exactly once, just starting mid-cycle.
* **Continual-learning metrics every round.** Each client's ``validate()`` runs a
  shared ``Evaluator`` (in ``scaleoututil.evaluation``) over its slice of every
  domain seen so far, emitting per-domain accuracy plus plasticity, forward
  and backward transfer, forgetting and average accuracy.
* **Programmatic launch.** Clients are not started via the CLI per terminal.
  Instead, ``run.py`` takes a single enrollment token and spawns ``N``
  ``scaleout client start`` subprocesses with shared partitioning parameters
  and a cross-process file lock that serializes training and validation so
  peak memory stays bounded under the ResNet-18 workload. Each subprocess
  exchanges the enrollment token for its own client credentials at startup.

**Note:** New to Scaleout Edge? Start with the MNIST quickstart:
https://docs.scaleoutsystems.com/en/latest/quickstart.html


Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__
-  ~5 GB free disk for the dataset on first run: the Avalanche download and
   its extracted images (~3.3 GB) under ``client/data/core50/`` plus the
   preprocessed per-domain tensors (~1.4 GB) under ``client/data/processed/``.


Creating the compute package and seed model
-------------------------------------------

Install ``scaleout`` and locate into this example directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client/scaleout-client-python/examples/core50-pytorch
   python -m venv .venv
   source .venv/bin/activate
   pip install scaleout

Login to Scaleout Edge:

.. code-block::

   scaleout login <URL>

Create the compute package:

.. code-block::

   scaleout package create --path client

Generate the seed model (ResNet-18, ImageNet pretrained, fresh 10-class head):

.. code-block::

   scaleout run install --path client
   scaleout run build --path client

``seed.npz`` (~45 MB) appears in the project root.


Running the clients
-------------------

Unlike the other examples, clients are launched programmatically. First mint
an enrollment token (one token enrolls all clients):

.. code-block::

   scaleout client create-enrollment-token --name core50

Then launch the clients with that token:

.. code-block::

   python run.py --api-url <URL> --enrollment-token <TOKEN> -n 4

**Non-TLS deployments.** Clients connect to the combiner over a TLS gRPC
channel by default. Against a local or otherwise plaintext deployment, export
``SCALEOUT_GRPC_SECURE=false`` before launching -- otherwise the handshake
fails with ``WRONG_VERSION_NUMBER``. See ``docs/client-envs.rst``.

This will:

1. Allocate a shared cross-process train/validate lock under ``$TMPDIR``.
2. Spawn ``N`` ``scaleout client start`` subprocesses, each with a unique
   ``CLIENT_NUMBER``, the shared ``TOTAL_N_CLIENTS``, and
   ``SCALEOUT_CLIENT_COMPUTE_LOCK`` in its environment. Each subprocess
   exchanges the enrollment token for its own credentials and connects as
   ``core50-client-<i>``.
3. Stream all per-client logs to stdout, prefixed by client number.

On the first run the clients will download CORe50 mini into
``client/data/core50/`` (idempotent; subsequent runs reuse the cache). The
8 training experiences are then materialized into per-domain tensors under
``client/data/processed/`` once and shared across clients.


How the data is partitioned
---------------------------

Each client receives ``CLIENT_NUMBER`` in ``[0, TOTAL_N_CLIENTS)`` and
``TOTAL_N_CLIENTS``. For every domain:

1. The domain's samples are split 80/20 into train/test using a deterministic,
   domain-stable seed.
2. The chosen half is shuffled by a separate (still domain-stable) seed.
3. The result is sliced into ``TOTAL_N_CLIENTS`` equal partitions; this client
   takes its index.

Two clients with the same numbers produce bit-identical slices without any
explicit coordination, and slices for distinct clients are disjoint.


Continual-learning metrics
--------------------------

``client.validate()`` calls ``Evaluator.evaluate(model)``, which runs the eval
function across every domain trained so far (using this client's test slices),
records the per-domain accuracy into an internal history matrix and emits:

* ``domain{i}_accuracy`` — accuracy on domain ``i``, for every domain trained
  so far. Only the evaluator's *primary* metric is surfaced per domain, so
  there is no ``domain{i}_loss``.
* ``avg_accuracy``, ``avg_loss`` — mean across all seen domains, one ``avg_``
  entry per metric the eval function returns.
* ``plasticity_accuracy`` — mean accuracy on each domain as measured right
  after that domain was trained.
* ``fwt_accuracy`` — forward transfer: mean accuracy on domains measured
  *before* they were trained.
* ``forgetting_accuracy`` — mean drop from each domain's peak past accuracy.
* ``bwt_accuracy`` — backward transfer: mean of current minus baseline accuracy
  on each past domain.
* ``domain_index`` — the domain this client trained on most recently.

The CL aggregates are suffixed with the evaluator's primary metric name, so a
different ``primary`` yields e.g. ``plasticity_f1``. ``forgetting_accuracy``
and ``bwt_accuracy`` only appear once at least one domain has a past round --
they are absent from the first round's payload.

Metrics are pushed via the standard ``client.log_metric`` channel and appear in
the Scaleout Edge dashboard alongside the usual training loss/accuracy.
