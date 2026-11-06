from typing import Optional

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg


class LoggingContext:
    """Context for keeping track of the session, model and round IDs during a call from a request."""

    def __init__(
        self,
        *,
        step: int = 0,
        model_id: str = None,
        round_id: str = None,
        round_number: int = None,
        session_id: str = None,
        request: Optional[scaleout_msg.TaskRequest] = None,
    ) -> None:
        if request is not None:
            if model_id is None:
                model_id = request.model_id
            if round_id is None:
                round_id = request.round_id
            if round_number is None:
                round_number = request.round_number
            if session_id is None:
                session_id = request.session_id

        self.model_id = model_id
        self.round_id = round_id
        # The round's ordinal within its session, 1-indexed. 0 (or None) when the task is
        # not tied to a numbered round, or the combiner predates the round_number field.
        self.round_number = round_number
        self.session_id = session_id
        self.request = request
        self.step = step
