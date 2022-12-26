from enum import Enum


class TaskType(Enum):
    classification = 0
    regression = 1

    @staticmethod
    def from_str(task_type: str):
        if task_type == "classification":
            return TaskType.classification
        elif task_type == "regression":
            return TaskType.regression
        else:
            raise ValueError("Invalid task type: {}".format(task_type))

    @staticmethod
    def list_str():
        return ["classification", "regression"]
