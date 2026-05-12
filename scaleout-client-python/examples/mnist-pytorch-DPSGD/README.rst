Scaleout Edge Project: Federated DP-SGD MNIST (Opacus + PyTorch)
-----------------------------------------------------------------

This Scaleout Edge example demonstrates how Differential Privacy can be integrated into a federated learning workflow to protect client data during training.

The project extends our baseline MNIST PyTorch example by incorporating the Opacus framework, which provides tools for training PyTorch models with Differential Privacy using DP-SGD.

To learn more about Differential Privacy in federated learning, see our blog post:
https://www.scaleoutsystems.com/post/guaranteeing-data-privacy-for-clients-in-federated-machine-learning

**Note:** We recommend that all new users start by taking the Quickstart Tutorial:
https://docs.scaleoutsystems.com/en/latest/quickstart.html


Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__

Notice
------

This implementation ensures that client data records used during training in a federated learning setting are protected using (epsilon, delta)-Differential Privacy.

The privacy budget (epsilon, delta) is applied independently per client, meaning that the privacy guarantee for one client is not affected by the privacy settings of other clients in the federation.

Since the total privacy budget is distributed across training rounds, the federation must agree on the expected number of rounds before training begins. The number of rounds must be provided to the Opacus privacy engine so it can determine an appropriate noise level for training.

Client-specific settings are provided through a ``client_settings.yaml`` file. The path to this file must be specified using the environment variable:

.. code-block::
   export CLIENT_SETTINGS_PATH=/path/to/client_settings.yaml

Edit Client-Specific Differential Privacy Parameters
----------------------------------------------------

The **Differential Privacy budget** (``epsilon``, ``delta``), along with other settings, is configurable in the ``client_settings.yaml`` file:

- **epsilon**: Total epsilon privacy budget to spend during training. The per-round privacy cost depends on the number of ``global_rounds`` configured on the server side.
- **delta**: Target delta value for the differential privacy guarantee.
- **accountant**: Privacy accountant method (e.g., ``rdp``, ``prv``, or ``gdp``).
- **max_grad_norm**: Gradient clipping threshold applied before adding noise.
- **batch_size**: Logical batch size (referred to as *lot size* in some literature).
- **optimizer**:
   - **name**: Optimizer method.
   - **kwargs**:
      - **lr**: Learning rate used by the optimizer.
      - **weight_decay**: Weight decay factor applied by the optimizer.
- **max_physical_batch_size**: Maximum batch size processed in memory during training. Larger logical batches may be split into smaller physical batches to reduce memory usage.
- **epochs**: Number of local training epochs performed by the client in each federated round.
- **global_rounds**: Expected number of federated training rounds run by the server. The per-round privacy spending is derived from ``epochs``, ``epsilon``, and ``global_rounds``.
- **hardlimit**:
   - If ``hardlimit`` is set to ``True``, the client will strictly enforce the epsilon budget and stop performing updates once the privacy budget has been exhausted.
   - If ``hardlimit`` is set to ``False``, the expected ``epsilon`` will approximately match the specified value, assuming the server completes the configured ``global_rounds``.

Creating the compute package and seed model
-------------------------------------------

Install scaleout:

Clone the Scaleout repository and locate into this example directory:

We recommend installing in a virtual environment.

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client/scaleout-client-python/examples/mnist-pytorch-DPSGD
   python -m venv .venv
   source .venv/bin/activate
   pip install scaleout


Login to Scaleout Edge:

Before running any commands, login to your Scaleout Edge instance:

.. code-block::

   scaleout login <URL>

Create the compute package:

.. code-block::

   scaleout package create --path client

This creates a file ``package.tgz`` in the project folder.


Next, generate a seed model (the first model in a global model trail).  
Install dependencies and build the client:

.. code-block::

   scaleout run install --path client
   scaleout run build --path client

This will create a model file ``seed.npz`` in the root of the project.  
This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).


Running the project on Scaleout Edge
---------------------------

To learn how to set up your Scaleout project and connect clients,
take the quickstart tutorial: https://docs.scaleoutsystems.com/en/latest/quickstart.html
