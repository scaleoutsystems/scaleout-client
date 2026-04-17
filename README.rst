|pic2|

.. |pic2| image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

Scaleout Edge: Sovereign Edge AI orchestration and Federated Learning
-------------------------------------------------------------------------------------- 

Scaleout Edge helps AI developers manage the machine learning lifecycle over distributed edge nodes and data silos. From a central control plane, operators can securely distribute models to edge nodes, monitor model metrics and telemetry on edge, fine tune models for local environments, and collaboratively train models over fleets of nodes.  

Built ground up around the concept of federated learning, the platform goes beyond edge inference by enabling secure, collaborative model training without requiring raw data to ever leave its source. This allows for scaling data access for machine learning over large volumes of sensitive data at the edge, and for speeding up the modeling loop by enabling adaptive AI and human-in-the-loop workflows.  

Key features include:

-  **Edge Model Operations** - Distribute, deploy and manage ML models and ML code on edge nodes for real-time inference, training and orchestration of ML workloads.

-  **Federated Learning** - Train models collaboratively across multiple edge nodes or data silos without sharing raw data.

-  **Central control over distributed ML workloads** - Manage and orchestrate distributed ML workloads from edge to cloud from a central control plane.

-  **Machine learning framework agnostic**. A flexible SDK (this repository) lets data scientists build their own clients and integrate with their ML framework of choice. 

-  **Infrastructure vendor-agnostic.** We ensure a wide range of deployment options including private cloud and on-premise infrastructure, and enable interfaces to leading cloud native software stacks. 

Getting started
============================

Get started with Scaleout Edge in two steps:  

1. Register for platform access at `Scaleout Edge Account <https://www.scaleoutsystems.com/pricing>`__
2. Take the `Quickstart tutorial <https://docs.scaleoutsystems.com/en/latest/quickstart.html>`__

Documentation
=============

More details about the architecture, deployment, and how to develop your own applications are found in the documentation:

-  `Documentation <https://docs.scaleoutsystems.com/en/latest>`__

Examples
=====================

Our example projects demonstrate different use case scenarios of Scaleout Edge 
and its integration with popular machine learning frameworks like PyTorch and TensorFlow.

- `Using the Python SDK <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/api-tutorials>`__
- `Federated Learning with PyTorch <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/mnist-pytorch>`__
- `Federated Learning with Tensforflow/Keras <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/mnist-keras>`__
- `Federated Learning with Hugging Face (Transformers) <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/huggingface>`__
- `Federated Learning with self-supervised Learning <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/FedSimSiam>`__
- `Federated Learning + Differential Privacy <https://github.com/scaleoutsystems/scaleout-client/tree/master/examples/mnist-pytorch-DPSGD>`__


Deployment Options
=================================

Several deployment and hosting options are available to suit different project requirements.

-   Dedicated cloud (single-tenant): Scaleout managed, dedicated deployment in a supported cloud VPC of your choice (GCP, Azure, OCI, AWS, Managed Kubernetes) 
-   Sovereign / self-managed: Deploy in your own VPC or on-premise Kubernets cluster using Scaleout's provided container images. 

For both hosted and self-managed deployments, tooling is availabale for all-in-one docker-compose deployments (starter/sandbox, single VM/host) and for scaling production deployments on Kubernetes.
Contact the Scaleout team for information.

Support
=================

Community support is available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Options are available for `Dedicated support <https://www.scaleoutsystems.com/start#pricing>`__.

Making contributions
====================

All pull requests will be considered and are much appreciated. For
more details please refer to our `contribution
guidelines <https://github.com/scaleoutsystems/scaleout-client/blob/master/CONTRIBUTING.md>`__.

Relationship to Scaleout FEDn
=============================

Scaleout Edge is the evolution of FEDn, the federated learning framework that Scaleout has been developing since 2019. FEDn now forms the federated learning engine
in Scaleout Edge. Core federated learning features include:

- Tiered federated learning architecture enabling massive scalability and resilience. 
- Support for any ML framework (examples for PyTorch, Tensforflow/Keras and Scikit-learn)
- Extendable via a plug-in architecture (aggregators, load balancers, object storage backends, databases  etc.)
- Built-in federated algorithms (FedAvg, FedAdam, FedYogi, FedAdaGrad, etc.)
- UI, CLI and Python API.
- Implement clients in any language (Python, C++, Kotlin etc.)
- No open ports needed client-side.


If you use Scaleout Edge for federated learning in your academic research, please consider citing:

::

   @article{ekmefjord2021scalable,
     title={Scalable federated machine learning with FEDn},
     author={Ekmefjord, Morgan and Ait-Mlouk, Addi and Alawadi, Sadi and {\AA}kesson, Mattias and Stoyanova, Desislava and Spjuth, Ola and Toor, Salman and Hellander, Andreas},
     journal={arXiv preprint arXiv:2103.00148},
     year={2021}
   }


License
=======

The Scaleout Edge SDK (this repository) is licensed under Apache-2.0 (see `LICENSE <LICENSE>`__ file for
full information).

Use of the Scaleout Edge platform is subject to the `Terms of Use <https://www.scaleoutsystems.com/terms>`__, see the Master Software License Agreement (MSLA) for full details.
