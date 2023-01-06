from dataclasses import dataclass
from typing import Callable, List, TypedDict, Any, Literal, Dict
from functools import partial
import copy
import numpy as np
from sklearn import metrics as skmetrics
from boost.problem import ProblemType

MetricsType = Dict[Literal['auc', 'logloss', 'f1', 'accuracy', 'mlogloss', 'r2', 'mse', 'mae', 'rmse', 'rmsle'], Any]


@dataclass
class Metrics:
    problem: ProblemType

    # This shouldn't be in `__post_init__`.
    # TODO: Move this in a callable function.
    def __post_init__(self):
        values: List[tuple[str, Callable]] = []

        match self.problem:
            case ProblemType.binary_classification:
                values.append(('auc', skmetrics.roc_auc_score))
                values.append(('logloss', skmetrics.log_loss))
                values.append(('f1', skmetrics.f1_score))
                values.append(('accuracy', skmetrics.accuracy_score))
                values.append(('precision', skmetrics.precision_score))
                values.append(('recall', skmetrics.recall_score))
            case ProblemType.multi_class_classification:
                values.append(('logloss', skmetrics.log_loss))
                values.append(('accuracy', skmetrics.accuracy_score))
                values.append(('mlogloss', skmetrics.log_loss))
            case [ProblemType.single_column_regression, ProblemType.multi_column_regression]:
                values.append(('r2', skmetrics.r2_score))
                values.append(('mse', skmetrics.mean_squared_error))
                values.append(('mae', skmetrics.mean_absolute_error))
                values.append(('rmse', partial(skmetrics.mean_squared_error, squared=False)))
                values.append(('rmsle', partial(skmetrics.mean_squared_log_error, squared=False)))
            case ProblemType.multi_label_classification:
                values.append(('logloss', skmetrics.log_loss))
            case _:
                raise Exception("Invalid problem type")

        self.values = values

    def calculate(self, y_true: List[Any], predictions: List[Any]) -> MetricsType:
        metrics: MetricsType = {}
        for name, func in self.values:
            if self.problem == ProblemType.binary_classification:
                if name == "auc":
                    metrics[name] = func(y_true, predictions[:, 1])
                if name == "logloss":
                    metrics[name] = func(y_true, predictions)
                else:
                    metrics[name] = func(y_true, predictions[:, 1] >= 0.5)
            elif self.problem == ProblemType.multi_class_classification:
                if name == "accuracy":
                    metrics[name] = func(y_true, np.argmax(predictions, axis=1))
                else:
                    metrics[name] = func(y_true, predictions)
            else:
                if name == "rmsle":
                    temp_predictions = copy.deepcopy(predictions)
                    temp_predictions[temp_predictions < 0] = 0
                    metrics[name] = func(predictions, temp_predictions)
                else:
                    metrics[name] = func(y_true, predictions)

        return metrics
