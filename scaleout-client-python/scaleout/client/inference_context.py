"""InferenceContext: carries inference run metadata for log_inference_result calls."""


class InferenceContext:
    """Data carrier for an active inference run.

    Mirrors LoggingContext but scoped to inference: holds inference_id and
    model_id, which are stamped on every InferenceResult enqueued during the run.
    """

    def __init__(self, *, inference_id: str, model_id: str) -> None:
        self.inference_id = inference_id
        self.model_id = model_id
