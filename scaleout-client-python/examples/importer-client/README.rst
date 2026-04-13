Scaleout Project: Importer Client
-----------------------------

This is an example Scaleout Project on how to design a client that imports client training code rather than running it in a separate process. 
This enables the user to have access to the grpc channel to send information to thecontroller during training. 

   **Note: We recommend that all new users start by taking the Quickstart Tutorial: https://docs.scaleoutsystems.com/en/latest/quickstart.html** 

Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__

Creating the compute package and seed model
-------------------------------------------

Clone the repository:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd examples/importer-client

Install scaleout in a new virtual environment:

.. code-block::

   python -m venv scaleout-env
   source scaleout-env/bin/activate
   pip install scaleout

Login to Scaleout Edge:

Before running any commands, login to your Scaleout Edge instance:

.. code-block::

   scaleout login <URL>

Install dependencies in the virtual environment:
.. code-block::

   scaleout run install --path client

Create the compute package:

.. code-block::

   scaleout package create --path client

This creates a file 'package.tgz' in the project folder.

Create a seed model file:
.. code-block::

   scaleout run build --path client

Upload the compute package and seed model to the Scaleout Controller:
.. code-block::

   scaleout package set-active --file package.tgz -n v1
   scaleout model set-active --file seed.npz

Running the project on Scaleout
----------------------------

.. code-block::

   scaleout client start --init client.yaml

To learn how to set up your Scaleout Studio project and connect clients, take the quickstart tutorial: https://docs.scaleoutsystems.com/en/latest/quickstart.html. 
