Multi-client enrollment
=======================

Run several edge clients concurrently from a **single shared enrollment token**.
Each client runs in its own thread, generates its own client id, and enrolls
independently — so one enrollment token can bootstrap a whole fleet.

This is the programmatic equivalent of running
``scaleout client start --token <enrollment-token>`` N times in parallel.

Prerequisites
-------------

* A running Scaleout deployment (API + combiner).
* A compute package the clients can load — either uploaded to the deployment
  (remote) or a local ``client/`` directory.
* An enrollment token.

Quick start
-----------

.. code-block:: bash

   # 1. Log in and mint a short-lived enrollment token
   scaleout login https://your-deployment
   export SCALEOUT_API_URL="https://your-deployment"
   export SCALEOUT_ENROLLMENT_TOKEN="$(scaleout client create-enrollment-token --name fleet --expires-in 1h)"

   # 2. (optional) pin a specific uploaded package; omit to use the session default
   # export SCALEOUT_PACKAGE_NAME="<uploaded-package-name>"

   # 3. Run three clients concurrently
   python run_clients.py --clients 3

Stop with ``Ctrl+C``.

Configuration
-------------

============================  ==========================================================
Environment variable          Meaning
============================  ==========================================================
``SCALEOUT_API_URL``          Deployment API URL (required).
``SCALEOUT_ENROLLMENT_TOKEN`` Shared enrollment token (required).
``SCALEOUT_PACKAGE``          ``remote`` (default) or ``local``.
``SCALEOUT_PACKAGE_NAME``     Optional; pin a specific uploaded package (default: session's active package).
``SCALEOUT_NUM_CLIENTS``      Default client count (overridden by ``--clients``).
============================  ==========================================================

Each client enrolls as a distinct ``client_id`` and manages its own access/refresh
tokens, refreshing automatically in the background. Revoking one enrolled client
(``POST /api/v1/clients/<client_id>/revoke``) stops only that client.
