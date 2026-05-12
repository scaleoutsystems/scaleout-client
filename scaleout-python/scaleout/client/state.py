"""Module for representing and converting client states."""

from enum import Enum


class ClientState(Enum):
    """Enum for representing the state of a client."""

    idle = 1
    training = 2
    validating = 3


def client_state_to_string(state: ClientState) -> str:
    """Convert a ClientState to a string representation.

    :param state: the state to convert
    :type state: :class:`scaleout.network.clients.state.ClientState`
    :return: string representation of the state
    :rtype: str
    """
    if state == ClientState.idle:
        return "IDLE"
    if state == ClientState.training:
        return "TRAINING"
    if state == ClientState.validating:
        return "VALIDATING"

    return "UNKNOWN"
