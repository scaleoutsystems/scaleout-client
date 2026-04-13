ASYNC CLIENTS 
-------------

This example shows how to experiment with intermittent and asynchronous client workflows.     

Prerequisites
-------------

- [Python 3.11, 3.12 or 3.13](https://www.python.org/downloads)
- [Docker (if running locally)](https://docs.docker.com/get-docker)
- [Docker Compose (if running locally)](https://docs.docker.com/compose/install)

Set up environment
------------------

First, make sure that Scaleout is installed (we recommend using a virtual environment)

Clone Scaleout

.. code-block::

    git clone https://github.com/scaleoutsystems/scaleout-client.git

Install Scaleout

.. code-block::

    pip install scaleout

Standing in the folder scaleout-client/scaleout-client-python/examples/async-clients

.. code-block::

    pip install -r requirements.txt

Upload seed model
------------

Create the seed model

.. code-block::

    python init_seed.py


You will now have a file 'seed.npz' in the directory. Add this seed model to the Scaleout instance:

.. code-block::

    python init_scaleout.py

Project configuration
------------

The file ``config.py`` contains all configuration settings for this example. The most important setting is ``USE_LOCAL`` which determines whether to run with a local Scaleout deployment or connect to a remote instance such as a Studio instance project.

**For local deployment:**
- Set ``USE_LOCAL = True`` in ``config.py``
- Deploy Scaleout locally using Docker Compose. Standing in the Scaleout root directory:

.. code-block::

    docker compose up 

**For remote deployment:**
- Set ``USE_LOCAL = False`` in ``config.py``
- Set up a project in Scaleout Studio
- Update the ``DISCOVER_HOST`` in ``REMOTE_CONFIG`` with your API URL
- Create a ``tokens.json`` file with the following structure:

.. code-block::

    {
        "api.scaleout.scaleoutsystems.com/your-project-name": {
            "CLIENT_TOKEN": "your-client-token-here",
            "ADMIN_TOKEN": "your-admin-token-here"
        }
    }

Replace ``your-project-name``, ``your-client-token-here``, and ``your-admin-token-here`` with your actual values.

Monitoring client status (optional)
----------------------------------

If you want to monitor client statuses, edit ``client_status.py`` and update the ``MACHINE_NAMES`` list with the names of the machines running your clients. Then run:

.. code-block::

    python client_status.py

This will periodically check and record client statuses to a CSV file.

Running clients and analyzing participation
------------------------------------------

Start simulating clients:

.. code-block::

    python run_clients.py

You can use the ``--intermittent`` flag to simulate clients that periodically disconnect and reconnect.

Start the experiment / training sessions: 

.. code-block::

    python run_experiment.py

You can adjust the number of sequential training sessions by modifying the ``N_SESSIONS`` parameter in ``config.py``. If you are using Scaleout Studio, you can also start a session directly through the Studio interface or use the Scaleout.

To analyze client participation and identify potential issues after a session has started:

.. code-block::

    python client_participation.py

This will generate plots showing the number of aggregated models and validations per round, helping you understand client participation patterns and identify where things might be going wrong.

Once global models start being produced, you can start analyzing results using API Client, refer to the notebook "Experiment.ipynb" for instructions.
