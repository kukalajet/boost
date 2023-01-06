import copy
from dataclasses import dataclass
from typing import Optional, Literal, Type, List, Dict, Any

from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from optuna import Trial, create_study
import numpy as np
import pandas as pd
from functools import partial

from boost.problem import ProblemType
from boost.metrics import Metrics
from boost.model import get_model_and_hyperparameters
from boost.logger import logger
import boost.fs as fs


class Learner:
    def __init__(
            self,
            problem_type: ProblemType,
            model_id: str,
            features: List[str],
            targets: List[str],
            num_folds: int,
            num_trials: int,
            has_tests: bool,
            time_limit: Optional[int],
            use_gpu: bool,
            seed: int,
            idx: str,
    ):
        self.problem_type = problem_type
        self.model_id = model_id
        self.features = features
        self.targets = targets
        self.num_folds = num_folds
        self.num_trials = num_trials
        self.has_tests = has_tests
        self.time_limit = time_limit
        self.use_gpu = use_gpu
        self.seed = seed
        self.idx = idx

    def _optimize(
            self,
            trial: Trial,
            eval_metric: Literal['logloss', 'mlogloss', 'rmse'],
            model_class: Type[XGBClassifier | XGBRegressor],
            predict_probabilities: bool,
    ):
        params = _get_params(trial, self.use_gpu)
        metrics = Metrics(self.problem_type)

        scores = []
        for fold in range(self.num_folds):
            train_fold_path = fs.get_model_fold_path(self.model_id, None, "train", fold)
            train_fold = pd.read_feather(train_fold_path)
            x_train = train_fold[self.features]
            y_train = train_fold[self.targets].values

            valid_fold_path = fs.get_model_fold_path(self.model_id, None, "valid", fold)
            valid_fold = pd.read_feather(valid_fold_path)
            x_valid = valid_fold[self.features]
            y_valid = valid_fold[self.targets].values

            model = _get_model_instance_with_params(model_class, params, eval_metric, random_state=self.seed)

            if self.problem_type in (
                    ProblemType.multi_column_regression, ProblemType.multi_label_classification):
                predictions = []
                models = [model] * len(self.targets)

                for index, current_model in enumerate(models):
                    current_model.fit(
                        x_train,
                        y_train[:, index],
                        eval_set=[(x_valid, y_valid[:, index])],
                        verbose=False,
                    )

                    if self.problem_type == ProblemType.multi_column_regression:
                        prediction = current_model.predict(x_valid)
                    else:
                        prediction = current_model.predict_proba(x_valid)[:, 1]
                    predictions.append(prediction)

                predictions = np.column_stack(predictions)
            else:
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_valid, y_valid)],
                    verbose=False
                )

                if predict_probabilities:
                    predictions = model.predict_proba(x_valid)
                else:
                    predictions = model.predict(x_valid)

            score = metrics.calculate(y_valid, predictions)
            scores.append(score)

        mean_metrics = _mean_dict_values(scores)
        _log_metrics(mean_metrics)

        return mean_metrics[eval_metric]

    def _predict(self, params: Dict[str, Any]):
        if self.use_gpu is True:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = 0
            params["predictor"] = "gpu_predictor"

        model_class, predict_probabilities, eval_metric, _ = get_model_and_hyperparameters(self.problem_type)
        metrics = Metrics(self.problem_type)
        target_encoder = _load_persisted_target_encoder(self.model_id, None)

        valid_predictions = {}
        test_predictions = []
        scores = []
        for fold in range(self.num_folds):
            logger.info(f"Training and predicting for fold {fold}")

            train_fold_path = fs.get_model_fold_path(self.model_id, None, "train", fold)
            train_fold = pd.read_feather(train_fold_path)
            x_train = train_fold[self.features]
            y_train = train_fold[self.targets].values

            valid_fold_path = fs.get_model_fold_path(self.model_id, None, "valid", fold)
            valid_fold = pd.read_feather(valid_fold_path)
            x_valid = valid_fold[self.features]
            y_valid = valid_fold[self.targets].values
            valid_ids = valid_fold[self.idx].values

            if self.has_tests:
                test_fold_path = fs.get_model_fold_path(self.model_id, None, "test", fold)
                test_fold = pd.read_feather(test_fold_path)
                x_test = test_fold[self.features]
                test_ids = test_fold[self.idx].values

            model = _get_model_instance_from_best_params(model_class, params, eval_metric, random_state=self.seed)

            if self.problem_type in (
                    ProblemType.multi_column_regression, ProblemType.multi_label_classification):
                predictions = []
                test_predictions = []
                models = []

                for index in range(len(self.targets)):
                    cloned_model = copy.deepcopy(model)
                    cloned_model.fit(x_train, y_train[:, index], eval_set=[(x_valid, y_valid[:, index])], verbose=False)
                    models.append(cloned_model)

                    if self.problem_type == ProblemType.multi_column_regression:
                        prediction = cloned_model.predict(x_valid)
                        if self.has_tests:
                            test_prediction = cloned_model.predict(x_test)
                    else:
                        prediction = cloned_model.predict_proba(x_valid)[:, 1]
                        if self.has_tests:
                            test_prediction = cloned_model.predict_proba(x_test)[:, 1]

                    predictions.append(prediction)
                    if self.has_tests:
                        test_predictions.append(test_prediction)

                predictions = np.column_stack(predictions)
                if self.has_tests:
                    test_predictions = np.column_stack(test_predictions)

                _save_model(models, self.model_id, None, fold)

            else:
                model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

                if predict_probabilities:
                    predictions = model.predict_proba(x_valid)
                    if self.has_tests:
                        test_predictions = model.predict_proba(x_test)
                else:
                    predictions = model.predict(x_valid)
                    if self.has_tests:
                        test_predictions = model.predict(x_test)

                _save_model(model, self.model_id, None, fold)

            valid_predictions.update(dict(zip(valid_ids, predictions)))
            if self.has_tests:
                test_predictions.append(test_predictions)

            metric = metrics.calculate(y_valid, predictions)
            scores.append(metric)
            logger.info(f"Fold {fold} done!")

        mean_metrics = _mean_dict_values(scores)
        _log_metrics(mean_metrics)
        _save_predictions(valid_predictions, self.idx, self.targets, self.model_id, target_encoder)

        if self.has_tests:
            _save_test_predictions(test_predictions, self.idx, self.targets, self.model_id, target_encoder, test_ids)
        else:
            logger.info("No test data supplied. Only OOF predictions were generated.")

    def _train(self):
        model_class, predict_probabilities, eval_metric, direction = get_model_and_hyperparameters(self.problem_type)
        optimize_function = partial(self._optimize, eval_metric=eval_metric, model_class=model_class,
                                    predict_probabilities=predict_probabilities)

        database_path = fs.get_optuna_database_path()
        study = create_study(direction=direction, study_name=self.model_id, storage=f"sqlite:///{database_path}",
                             load_if_exists=True)

        study.optimize(optimize_function, n_trials=self.num_trials, timeout=self.time_limit)

        return study.best_params

    def train(self):
        logger.info("Training started")
        params = self._train()
        logger.info("Training complete")

        logger.info("Creating OOF and test predictions")
        self._predict(params)
        logger.info("Creating OOF and test predictions complete")


def _log_metrics(metrics: Dict[str, float]):
    values = []
    for key, value in metrics.items():
        values.append(f"{key}: {value}")
    log = " | ".join(values)
    logger.info(log)


@dataclass(order=False)
class Params:
    learning_rate: float
    reg_lambda: float
    reg_alpha: float
    subsample: float
    colsample_bytree: float
    max_depth: int
    early_stopping_rounds: int
    n_estimators: int
    tree_method: Optional[Literal["gpu_hist", "exact", "approx", "hist"]] = None
    gpu_id: Optional[int] = None
    predictor: Optional[str] = None
    booster: Optional[Literal["gbtree", "gblinear"]] = None
    gamma: Optional[float] = None
    grow_policy: Optional[Literal["depthwise", "lossguide"]] = None


def _get_params(trial: Trial, use_gpu: bool) -> Params:
    params = Params(
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        subsample=trial.suggest_float("subsample", 0.1, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        max_depth=trial.suggest_int("max_depth", 1, 9),
        early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 100, 500),
        n_estimators=trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
    )

    if use_gpu:
        params.tree_method = "gpu_hist"
        params.gpu_id = 0
        params.predictor = "gpu_predictor"
    else:
        params.tree_method = trial.suggest_categorical("tree_method", ["exact", "approx", "hist"])
        params.booster = trial.suggest_categorical("booster", ["gbtree", "gblinear"])
        if params.booster == "gbtree":
            params.gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params.grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    return params


def _mean_dict_values(values: List[Dict[str, int]]) -> Dict[str, Any]:
    mean_dict = {}
    for key in values[0].keys():
        mean_dict[key] = sum(d[key] for d in values) / len(values)

    return mean_dict


def _get_model_instance_with_params(
        model: Type[XGBClassifier | XGBRegressor],
        params: Params,
        eval_metric: Literal['logloss', 'mlogloss', 'rmse'],
        random_state: int,
) -> XGBClassifier | XGBRegressor:
    return model(
        learning_rate=params.learning_rate,
        reg_lambda=params.reg_lambda,
        reg_alpha=params.reg_alpha,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        max_depth=params.max_depth,
        early_stopping_rounds=params.early_stopping_rounds,
        n_estimators=params.n_estimators,
        tree_method=params.tree_method,
        gpu_id=params.gpu_id,
        predictor=params.predictor,
        booster=params.booster,
        gamma=params.gamma,
        grow_policy=params.grow_policy,
        eval_metric=eval_metric,
        random_state=random_state,
        use_label_encoder=False,
    )


def _load_persisted_target_encoder(model_id: str, model_version: Optional[fs.DatasetVersion]) -> LabelEncoder:
    filename = f"{model_id}.target_encoder"
    target_encoder = fs.load_object(model_id, model_version, filename)
    return target_encoder


def _save_model(model: Any, model_id: str, model_version: Optional[fs.DatasetVersion], fold: int):
    filename = f"{model_id}-model.{fold}"
    fs.save_object(model, model_id, filename, model_version)


def _get_model_instance_from_best_params(
        model: Type[XGBClassifier | XGBRegressor],
        best_params: Dict[str, Any],
        eval_metric: Literal['logloss', 'mlogloss', 'rmse'],
        random_state: int,
) -> XGBClassifier | XGBRegressor:
    return model(random_state=random_state, eval_metric=eval_metric, use_label_encoder=False, **best_params)


def _save_predictions(
        predictions: Dict[int, np.ndarray],
        idx: str,
        targets: List[str],
        model_id: str,
        target_encoder: LabelEncoder | OrdinalEncoder
):
    predictions = pd.DataFrame.from_dict(predictions, orient="index").reset_index()
    if target_encoder is None:
        predictions.columns = [idx] + targets
    else:
        predictions.columns = [idx] + list(target_encoder.classes_)

    predictions_path = fs.get_predictions_path(model_id, None, "valid")
    predictions.to_csv(predictions_path, index=False)


def _save_test_predictions(
        predictions: np.ndarray,
        idx: str,
        targets: List[str],
        model_id: str,
        target_encoder: LabelEncoder | OrdinalEncoder,
        test_ids: List[int],
):
    predictions = np.mean(predictions, axis=0)
    if target_encoder is None:
        predictions = pd.DataFrame(predictions, columns=targets)
    else:
        predictions = pd.DataFrame(predictions, columns=list(target_encoder.classes_))
    predictions.insert(loc=0, column=idx, value=test_ids)

    predictions_path = fs.get_predictions_path(model_id, None, "test")
    predictions.to_csv(predictions_path, index=False)
