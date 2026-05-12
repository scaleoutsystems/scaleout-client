Scaleout Edge Project: Server functions
---------------------------------------

This example demonstrates how to use custom server functions (in ``server_functions.py``) to:

- **Leverage client attributes** to select specific clients for training (see ``startup.py`` and ``server_functions.py``).
- **Send dynamic, customizable payloads** from the server to clients via Python dictionaries.
- **Implement custom aggregation logic**.

Additionally, for large-scale experiments, ``sf_incremental_aggregation.py`` demonstrates
**memory-safe incremental averaging** using server functions.

For details on the functionality of server-functions see either the file server_functions.py,
the docs https://docs.scaleoutsystems.com/en/latest/serverfunctions.html or the youtube video
https://www.youtube.com/watch?v=Rnfhfqy_Tts.

**Note:** We recommend that all new users start by taking the Quickstart Tutorial:
https://docs.scaleoutsystems.com/en/latest/quickstart.html

Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__

To apply server-functionality in Scaleout Edge first connect to your project through the Scaleout.

See https://docs.scaleoutsystems.com/en/latest/apiclient.html for more information.

When connected to the project API you can start sessions with your supplied server functions.

Full commands to run through the API client:

Get your token from the settings page in your Scaleout project and add it in your system environment.

.. code-block::

    export SCALEOUT_AUTH_TOKEN=<access token>

Connect through the Scaleout from a python instance, you can find your controller host on the Scaleout Dashboard page.

.. code-block::

    from scaleout import Scaleout
    client = Scaleout(host="<controller-host>", secure=True, verify=True)

Start a session with your ServerFunctions code (assuming you have uploaded a model seed, compute package and have connected clients).

.. code-block::

    from server_functions import ServerFunctions
    client.start_session(server_functions=ServerFunctions)

Logs from the server functions code are visible from the Scaleout dashboard logs page.
