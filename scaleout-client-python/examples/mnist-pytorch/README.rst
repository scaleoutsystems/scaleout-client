Scaleout Edge Project: MNIST (PyTorch)
-----------------------------

This is a minimal Scaleout Edge project based on the classic hand-written digit recognition dataset MNIST,
implemented in PyTorch. The example is intended as a simple quickstart to learn how to use Scaleout Edge.

**Note:** We recommend that all new users start by taking the Quickstart Tutorial:
https://docs.scaleoutsystems.com/en/latest/quickstart.html


Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__


Creating the compute package and seed model
------------------------------------------

Install scaleout:

Clone the Scaleout repository and locate into this example directory:

We recommend installing in a virtual environment.

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client/scaleout-client-python/examples/mnist-pytorch
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
