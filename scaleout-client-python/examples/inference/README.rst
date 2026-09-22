Scaleout Edge Project: Inference
---------------------------------

This example demonstrates how to serve a model for **live inference** on an edge client, using
``stage_model_callback`` and ``inference_callback`` (see ``startup.py``):

- **Stage a model** for inference on the client, and track which models are currently cached
  locally via ``client.local_repository.models``.
- **Run inference** on incoming input data and stream results back with ``client.log_telemetry(...)``.

``build.py`` generates a minimal random-weights seed model so the example runs without any
training data or real model.

**Note:** We recommend that all new users start by taking the Quickstart Tutorial:
https://docs.scaleoutsystems.com/en/latest/quickstart.html

For the full ``EdgeClient`` API used here (callback registration, logging, and the inference
helpers), see https://docs.scaleoutsystems.com/en/latest/edge_client_sdk.html.

Triggering inference from the API client
-----------------------------------------

Assuming you have uploaded a model seed, built the compute package, and have connected clients
(see the quickstart tutorial above), you can stage a model and start inference through the
Scaleout API client:

.. code-block:: python

    from scaleout import Scaleout

    api_client = Scaleout(host="<deployment-url>", secure=True, verify=True)

    # Stage the model on the client(s) - triggers stage_model_callback
    api_client.stage_model("<model_id>")

    # Start an inference trail for the staged model - triggers inference_callback
    api_client.start_inference("<model_id>")

By default both calls target all connected clients; pass ``client_ids=[...]`` to either call to
target specific clients instead.

See https://docs.scaleoutsystems.com/en/latest/apiclient.html for more information on connecting
through the API client, including how to obtain and set your ``SCALEOUT_AUTH_TOKEN``.

