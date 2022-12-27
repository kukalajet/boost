import os
from typing import List, Any, Dict, Literal, Type, Optional, TypedDict
from pydantic import BaseModel
from functools import partial
from optuna import Trial, create_study
from numpy import np
from pandas import pd
from xgboost import XGBClassifier, XGBRegressor
from problem import ProblemType
from utils import dict_mean
from metrics import Metrics


class ModelConfig(BaseModel):
    train_filename: str
    test_filename: Optional[str] = None
    idx: str
    targets: List[str]
    problem: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool


ParamsDict = TypedDict('ParamsDict', {
    'learning_rate': float,
    'reg_lambda': float,
    'reg_alpha': float,
    'subsample': float,
    'colsample_bytree': float,
    'max_depth': int,
    'early_stopping_rounds': int,
    'n_estimators': Any,
    'tree_method': Any,
    'gpu_id': int,
    'predictor': Any,
    'booster': Any,
    'gamma': float,
    'grow_policy': Any,
})


def get_params(trial: Type[Trial], model_config: Type[ModelConfig]) -> ParamsDict:
    params: ParamsDict = {}
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)
    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
    params["early_stopping_rounds"] = trial.suggest_int("early_stopping_rounds", 100, 500)
    params["n_estimators"] = trial.suggest_categorical("n_estimators", [7000, 15000, 20000])

    if model_config.use_gpu:
        params["tree_method"] = "gpu_hist"
        params["gpu_id"] = 0
        params["predictor"] = "gpu_predictor"
    else:
        params["tree_method"] = trial.suggest_categorical("tree_method", ["exact", "approx", "hist"])
        params["booster"] = trial.suggest_categorical("booster", ["gbtree", "gblinear"])
        if params["booster"] == "gbtree":
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    return params


def _fetch_model_params(model_config: ModelConfig):
    problem = model_config.problem
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


def optimize(
        trial: any,
        xgb_model: Type[XGBClassifier] | Type[XGBRegressor],
        predict_probabilities: bool,
        eval_metric: Literal['logloss', 'mlogloss', 'rmse'],
        model_config: Type[ModelConfig],
) -> Dict[str, Any]:
    params = get_params(trial, model_config)
    early_stopping_rounds = params["early_stopping_rounds"]
    params["early_stopping_rounds"] = None

    metrics = Metrics(model_config.problem)

    scores = []

    for fold in range(model_config.num_folds):
        train_path = f"./output/train_fold_{fold}.feather"
        valid_path = f"./output/valid_fold_{fold}.feather"

        train_feather = pd.read_feather(train_path)
        valid_feather = pd.read_feather(valid_path)

        x_train = train_feather[model_config.features]
        x_valid = valid_feather[model_config.features]

        y_train = train_feather[model_config.targets].values
        y_valid = valid_feather[model_config.targets].values

        # train model
        model = xgb_model(
            random_state=model_config.seed,
            eval_metric=eval_metric,
            use_label_encoder=False,
            **params,
        )

        if model_config.problem in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
            predictions = []
            models = [model] * len(model_config.targets)

            for idx, _model in enumerate(models):
                _model.fit(
                    x_train,
                    y_train[:, idx],
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(x_valid, y_valid[:, idx])],
                    verbose=False,
                )

                if model_config.problem == ProblemType.multi_column_regression:
                    temp = _model.predict(x_valid)
                else:
                    temp = _model.predict_proba(x_valid)[:, 1]
                predictions.append(temp)

            predictions = np.column_stack(predictions)

        else:
            model.fit(
                x_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )

            if predict_probabilities:
                predictions = model.predict_proba(x_valid)
            else:
                predictions = model.predict(x_valid)

        score = metrics.calculate(y_valid, predictions)
        scores.append(score)
        if model_config.fast is True:
            break

    mean_metrics = dict_mean(scores)
    print(f"Metrics: {mean_metrics}")

    return mean_metrics[eval_metric]


def train_model(model_config: ModelConfig) -> Dict[str, Any]:
    model, predict_probabilities, eval_metric, direction = _fetch_model_params(model_config)

    optimize_function = partial(
        optimize,
        xgb_model=model,
        predict_probabilities=predict_probabilities,
        eval_metric=eval_metric,
        model_config=model_config
    )

    db_path = os.path.join(model_config.output, "params.db")
    study = create_study(
        direction=direction,
        study_name="testtesttest",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

    study.optimize(
        optimize_function,
        n_trials=model_config.num_trials,
        timeout=model_config.time_limit,
    )

    return study.best_params
