from typing import List, Literal, Optional, Type

from pydantic import BaseModel
from xgboost import XGBClassifier, XGBRegressor

from boost.problem import ProblemType


class ModelConfig(BaseModel):
    model_folder: str
    train_filename: str
    test_filename: Optional[str] = None
    problem_type: ProblemType
    idx: str
    targets: List[str]
    features: List[str]
    categorical_features: List[str]
    use_gpu: bool
    num_folds: int
    seed: int
    num_trials: int
    time_limit: Optional[int] = None


def _fetch_model_params(model_config: ModelConfig) -> (
        Type[XGBClassifier | XGBRegressor], bool, Literal["logloss", "mlogloss", "rmse"], str):
    problem = model_config.problem_type
    direction = "minimize"
    match problem:
        case ProblemType.binary_classification | ProblemType.multi_label_classification:
            model = XGBClassifier
            predict_probabilities = True
            eval_metric = "logloss"
        case ProblemType.multi_class_classification:
            model = XGBClassifier
            predict_probabilities = True
            eval_metric = "mlogloss"
        case ProblemType.single_column_regression | ProblemType.multi_column_regression:
            model = XGBRegressor
            predict_probabilities = False
            eval_metric = "rmse"
        case _:
            raise NotImplementedError

    return model, predict_probabilities, eval_metric, direction
