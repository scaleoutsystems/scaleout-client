from enum import Enum


class TaskType(Enum):
    ModelUpdate = "Scaleout_ModelUpdate"
    Validation = "Scaleout_Validation"

    StageModel = "Scaleout_StageModel"
    Inference = "Scaleout_Inference"

    @staticmethod
    def is_valid_task(task: str) -> bool:
        """Validate that the task is a valid task.

        :param task: The task to validate.
        :return: True if the task is valid, False otherwise.
        """
        if isinstance(task, TaskType):
            return True
        if not isinstance(task, str):
            return False
        if task.startswith("Custom_"):
            return True
        if task in [item.value for item in TaskType]:
            return True
        return False

    @staticmethod
    def is_scaleout_task(task: str) -> bool:
        """Check if the task is a Scaleout task.

        :param task: The task to check.
        :return: True if the task is a Scaleout task, False otherwise.
        """
        if isinstance(task, TaskType):
            return True
        if not isinstance(task, str):
            return False
        return task in [item.value for item in TaskType]

    @staticmethod
    def is_custom_task(task: str) -> bool:
        """Check if the task is a custom task.

        :param task: The task to check.
        :return: True if the task is a custom task, False otherwise.
        """
        if not isinstance(task, str):
            return False
        return task.startswith("Custom_")
