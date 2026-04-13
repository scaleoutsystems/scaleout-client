Scaleout Project: Keyword Detection (PyTorch)
-----------------------------

This is an example to showcase how to set up EdgeClient and use Scaleout to setup and manage a training from python. 
The machine learning project is based on the Speech Commands dataset from Google, https://huggingface.co/datasets/google/speech_commands.

The example is intented as a minimalistic quickstart to learn how to use Scaleout Edge.


   **Note: It is recommended to complete the example in https://docs.scaleoutsystems.com/en/latest/quickstart.html before starting this example ** 

Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__

Installing pre requirements and creating seed model
-------------------------------------------

There are two alternatives to install the required packages, either using conda or pip.

.. code-block::

   conda env create -n <name-of-env> --file env.yaml

Or if you rather use pip to install the packages:

.. code-block::

   pip install -r requirements.txt


Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client/scaleout-client-python/examples/pytorch-keyworddetection-api

Next we need to setup the Scaleout. This link https://docs.scaleoutsystems.com/en/latest/apiclient.html helps you to get the hostname and access token. Edit the file scaleout_api.py and insert your HOST and TOKEN.

Next, generate the seed model:

.. code-block::

   python scaleout_api.py --upload-seed

This will create a model file 'seed.npz' in the root of the project and upload it to the server.

Now we need to start the clients, download at set of client configutations following the quickstart tutorial: https://docs.scaleoutsystems.com/en/latest/quickstart.html#start-clients. 

Start the clients with the following command:
.. code-block::

   python main.py --dataset-split-idx 0 --client-yaml client.yaml

where each client is started with a different dataset split index and client yaml file.

