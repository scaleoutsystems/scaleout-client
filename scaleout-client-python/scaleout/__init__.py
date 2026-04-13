"""The Scaleout client package is responsible for executing the federated learning tasks, including ML model training and validation. It's the acting gRPC client for the federated network.
The client first connects the centralized controller to receive :class:`scaleout.network.combiner.Combiner` assignment. The client then connects to the combiner and
sends requests to the combiner to receive model updates and send model updates.
"""
# ruff: noqa: E501

from scaleout.client.edge_client import EdgeClient
from scaleoututil.api.client import Scaleout
from scaleoututil.utils.model import ScaleoutModel

__all__ = ["Scaleout", "EdgeClient", "ScaleoutModel"]
