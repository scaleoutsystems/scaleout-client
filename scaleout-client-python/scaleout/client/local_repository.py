from pathlib import Path
from typing import List

from scaleoututil.utils.model import ScaleoutModel


class LocalModelRepository:
    """A local model repository for storing and retrieving models."""

    def __init__(self, repository_path: str):
        self.repository_path = repository_path
        if not Path(self.repository_path).exists():
            Path(self.repository_path).mkdir(parents=True, exist_ok=True)
        self._models_cache: List[ScaleoutModel] = []

    @property
    def models(self) -> list[ScaleoutModel]:
        """List all models in the local repository.

        :return: A list of model names.
        """
        model_files = list(Path(self.repository_path).glob("*.scm"))

        model_cache = [model.model_id for model in self._models_cache]
        for model_file in model_files:
            model_id = model_file.stem
            if model_id not in model_cache:
                model = ScaleoutModel.from_file(model_file)
                if model_id != model.model_id:
                    raise ValueError(f"Model ID mismatch: expected {model_id}, got {model.model_id}")
                self._models_cache.append(model)

        # Remove models from cache that no longer exist in the repository
        for cached_model in list(self._models_cache):
            if not any(cached_model.model_id == model_file.stem for model_file in model_files):
                self._models_cache.remove(cached_model)

        return self._models_cache

    def stage_model(self, model: ScaleoutModel) -> None:
        """Stage a model for use.

        :param model: The ScaleoutModel to stage.
        """
        if self.model_exists(model):
            return
        model.save_to_file(Path(self.repository_path) / f"{model.model_id}.scm")
        self._models_cache.append(model)
        return

    def get_model_by_id(self, model_id: str) -> ScaleoutModel:
        """Get a model by id.

        :param model_name: The name of the model to retrieve.
        :return: The ScaleoutModel with the specified name.
        """
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def delete_model(self, model_id: str) -> None:
        """Delete a model from the repository.

        :param model_id: The id of the model to delete.
        """
        model_path = Path(self.repository_path) / f"{model_id}.scm"
        if model_path.exists():
            model_path.unlink()
            self._models_cache = [m for m in self._models_cache if m.model_id != model_id]

    def clear_repository(self) -> None:
        """Clear all models from the repository."""
        for model in Path(self.repository_path).glob("*.scm"):
            model.unlink()
        self._models_cache = []

    def model_exists(self, model: str | ScaleoutModel) -> bool:
        """Check if a model exists in the repository.

        :param model: The model id (str) or ScaleoutModel to check.
        :return: True if the model exists, False otherwise.
        """
        if isinstance(model, ScaleoutModel):
            model_id = model.model_id
        else:
            model_id = model
        return any(model.model_id == model_id for model in self.models)
