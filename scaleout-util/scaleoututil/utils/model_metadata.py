"""Carry the metadata of the model a round starts from over to the model the round aggregates.

Aggregates are built from weights alone, so metadata that clients put on their updates never reaches them.
"""

from scaleoututil.utils.model import ScaleoutModel

# Keys that describe a model's own content rather than what the model is; never carried over.
_OWN_KEYS = ("model_id", "helper_type", "inference_model_format")


def carried_over_metadata(model: ScaleoutModel) -> dict:
    """Return the metadata of a round's input model that the round's aggregate should carry."""
    return {k: v for k, v in model.metadata.items() if k not in _OWN_KEYS}


def with_metadata(model: ScaleoutModel, metadata: dict) -> ScaleoutModel:
    """Return the aggregated model with the given metadata added, closing the original when a new model is built."""
    if not metadata:
        return model
    result = model.to_builder().set_metadata_dict(metadata).build()
    model.close()
    return result
