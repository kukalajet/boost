from enum import IntEnum
from typing import List
from pandas import DataFrame
import numpy as np
from sklearn.utils.multiclass import type_of_target


class ProblemType(IntEnum):
    binary_classification = 1
    multi_class_classification = 2
    multi_label_classification = 3
    single_column_regression = 4
    multi_column_regression = 5

    @staticmethod
    def from_str(label):
        if label == "binary_classification":
            return ProblemType.binary_classification
        elif label == "multi_class_classification":
            return ProblemType.multi_class_classification
        elif label == "multi_label_classification":
            return ProblemType.multi_label_classification
        elif label == "single_column_regression":
            return ProblemType.single_column_regression
        elif label == "multi_column_regression":
            return ProblemType.multi_column_regression
        else:
            raise NotImplementedError


def determine_problem_type(
        targets: List[str],
        df: DataFrame,
        task: str = None,
) -> ProblemType:
    values = df[targets].values

    if task is None:
        target_type = type_of_target(values)

        if target_type == "continuous":
            problem = ProblemType.single_column_regression
        elif target_type == "continuous-multioutput":
            problem = ProblemType.multi_column_regression
        elif target_type == "binary":
            problem = ProblemType.binary_classification
        elif target_type == "multiclass":
            problem = ProblemType.multi_class_classification
        elif target_type == "multilabel-indicator":
            problem = ProblemType.multi_label_classification
        else:
            raise Exception("Unable to infer `problem`. Please provide `classification` or `regression`")

        return problem

    if task == "classification":
        if len(targets) == 1:
            unique_values = np.unique(values)
            if len(unique_values) == 2:
                problem = ProblemType.binary_classification
            else:
                problem = ProblemType.multi_label_classification
        else:
            problem = ProblemType.multi_label_classification

    elif task == "regression":
        if len(targets) == 1:
            problem = ProblemType.single_column_regression
        else:
            problem = ProblemType.multi_column_regression
    else:
        raise Exception("Problem type not understood")

    return problem
