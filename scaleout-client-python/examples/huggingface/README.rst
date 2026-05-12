Scaleout Edge Project: Hugging Face
-----------------------------------

This is an example project that demonstrates how one can make use of the Hugging Face Transformers library in Scaleout Edge.
In this example, a pre-trained BERT-tiny model from Hugging Face is fine-tuned to perform spam detection 
on the Enron spam email dataset.

Email communication often contains personal and sensitive information, and privacy regulations make it 
impossible to collect the data to a central storage for model training.
Federated learning is a privacy preserving machine learning technique that enables the training of models on decentralized data sources.
Fine-tuning large language models (LLMs) on various data sources enhances both accuracy and generalizability.
In this example, the Enron email spam dataset is split among two clients. The BERT-tiny model is fine-tuned on the client data using 
federated learning to predict whether an email is spam or not.

The user interface visualizes the training progress by plotting test loss and accuracy, as shown in the plot below. 
After running the example for only a few rounds in Scaleout Edge, the BERT-tiny model - fine-tuned via federated learning - 
is able to detect spam emails on the test dataset with high accuracy. 

.. image:: figs/hf_figure.png
   :width: 50%

To run the example, follow the steps below. For a more detailed explanation, follow the documentation.

**Note:** We recommend that all new users start by taking the Quickstart Tutorial:
https://docs.scaleoutsystems.com/en/latest/quickstart.html

Prerequisites
-------------

-  `Python >=3.11, <3.14 <https://www.python.org/downloads>`__


Creating the compute package and seed model
-------------------------------------------

Install scaleout: 

Clone the Scaleout repository and locate into this example directory:

We recommend installing in a virtual environment.

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client/scaleout-client-python/examples/huggingface
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
