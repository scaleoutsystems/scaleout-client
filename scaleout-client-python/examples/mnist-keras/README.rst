Scaleout Edge Project: Keras/Tensorflow (MNIST) 
-------------------------------------------

Prerequisites
-------------------------------------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__

Creating the compute package and seed model
-------------------------------------------

Install scaleout: 

recommend using a virtual environmnet
.. code-block::
   python -m venv venv
.. code-block::
.. code-block::

  git clone https://github.com/scaleoutsystems/scaleout-client.git
  cd scaleout-client/scaleout-client-python/examples/mnist-keras
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

This creates a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail). Now we need to install dependencies and build the client:

.. code-block::
   scaleout run install --path client
   scaleout run build --path client

This will create a model file 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

