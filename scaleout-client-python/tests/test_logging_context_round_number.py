"""LoggingContext exposes the round ordinal the combiner sent.

Uses the real generated TaskRequest rather than a stand-in, so this also pins
that round_number survives a serialize/parse round trip on the wire.
"""

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg

from scaleout.client.logging_context import LoggingContext

ROUND_UUID = "2f1a9c4e-0b6d-4a1e-9f3b-77c0d2a4e8b1"


def _on_the_wire(request: scaleout_msg.TaskRequest) -> scaleout_msg.TaskRequest:
    received = scaleout_msg.TaskRequest()
    received.ParseFromString(request.SerializeToString())
    return received


def test_round_number_survives_the_wire_and_reaches_the_context():
    request = scaleout_msg.TaskRequest(session_id="session-1", model_id="model-1", round_id=ROUND_UUID, round_number=4)

    context = LoggingContext(request=_on_the_wire(request))

    assert context.round_number == 4
    assert context.round_id == ROUND_UUID


def test_an_unset_round_number_reads_as_zero():
    """What a validation task, a custom command, or a combiner older than the field sends."""
    request = scaleout_msg.TaskRequest(session_id="session-1", model_id="model-1")

    context = LoggingContext(request=_on_the_wire(request))

    assert context.round_number == 0


def test_an_explicit_round_number_wins_over_the_request():
    context = LoggingContext(request=scaleout_msg.TaskRequest(round_number=4), round_number=9)

    assert context.round_number == 9
