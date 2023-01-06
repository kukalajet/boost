from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from boost.learner import Learner
from boost.logger import logger
from boost.problem import ProblemType, get_problem_type
from boost.model import ModelConfig
from boost.utils import reduce_memory_usage
import boost.fs as fs


class DataLoader:
    def __init__(
            self,
            model_id: str,
            task: Optional[str] = None,
            idx: Optional[str] = "id",
            targets: Optional[List[str]] = None,
            features: Optional[List[str]] = None,
            categorical_features: Optional[List[str]] = None,
            use_gpu: Optional[bool] = False,
            num_folds: Optional[int] = 5,
            seed: Optional[int] = 42,
            num_trials: Optional[int] = 1000,
            time_limit: Optional[int] = None,
    ):
        self.model_id = model_id
        self.task = task
        self.idx = idx
        self.targets = targets
        self.features = features
        self.categorical_features = categorical_features
        self.use_gpu = use_gpu
        self.num_folds = num_folds
        self.seed = seed
        self.num_trials = num_trials
        self.time_limit = time_limit

        if self.targets is None:
            logger.warning("No target columns specified. Will default to `target`.")
            self.targets = ["target"]
        if self.idx is None:
            logger.warning("No id column specified. Will default to `id`.")
            self.idx = "id"

    def _process_data(self):
        train_df = _get_processed_df_from_csv(self.model_id, "train", self.idx)
        test_df = _get_processed_df_from_csv(self.model_id, "test", self.idx)
        self.has_tests = True if test_df is not None else False

        problem_type = get_problem_type(self.targets, train_df, self.task)
        train_df = self._create_folds(train_df, problem_type)
        target_encoder, train_df = self._get_target_encoder(train_df, problem_type)

        self.features = self._get_features(train_df)
        categorical_features = self._get_categorical_features(train_df, self.features)

        logger.info(f"Found {len(categorical_features)} categorical features.")
        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")

        categorical_encoders = self._get_categorical_encoders(train_df, test_df, categorical_features)
        self.model_config = self._get_model_config(self.features, categorical_features, problem_type)

        _save_model_config(self.model_config, self.model_id, None)
        _save_categorical_encoders(categorical_encoders, self.model_id, None)
        _save_target_encoder(target_encoder, self.model_id, None)

    def _create_folds(self, train_df: pd.DataFrame, problem: ProblemType) -> pd.DataFrame:
        if "kfold" in train_df.columns:
            self.num_folds = len(np.unique(train_df["kfold"]))
            logger.info("Using `kfold` for folds from training data")
            return train_df

        train_df = _create_folds(train_df, self.targets, self.num_folds, problem, self.seed)
        return train_df

    def _get_model_config(
            self,
            features: List[str],
            categorical_features: List[str],
            problem_type: ProblemType
    ) -> ModelConfig:
        model_config = ModelConfig(
            model_id=self.model_id,
            problem_type=problem_type,
            idx=self.idx,
            targets=self.targets,
            features=features,
            categorical_features=categorical_features,
            use_gpu=self.use_gpu,
            num_folds=self.num_folds,
            seed=self.seed,
            num_trials=self.num_trials,
            time_limit=self.time_limit,
        )

        return model_config

    def _get_target_encoder(
            self,
            df: pd.DataFrame,
            problem_type: ProblemType
    ) -> (LabelEncoder | None, pd.DataFrame):
        # I hate this code but couldn't do much better without impacting too much the existing code.
        # TODO: Refactor this in a way that is not necessary to return a tuple.
        if problem_type not in [ProblemType.binary_classification, ProblemType.multi_class_classification]:
            return None, df

        target_encoder = LabelEncoder()
        target_values = df[self.targets].values
        target_encoder.fit(target_values.reshape(-1))
        df.loc[:, self.targets] = target_encoder.transform(
            target_values.reshape(-1))

        return target_encoder, df

    def _get_features(self, train_df: pd.DataFrame) -> List[str]:
        if self.features is not None:
            return self.features

        ignore_columns = [self.idx, "kfold"] + self.targets
        features = list(train_df.columns)
        features = [x for x in features if x not in ignore_columns]
        return features

    def _get_categorical_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        if self.categorical_features is not None:
            return self.categorical_features

        categorical_features = []
        for feature in features:
            if df[feature].dtype == "object":
                categorical_features.append(feature)

        return categorical_features

    def _get_categorical_encoders(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame | None,
            categorical_features: List[str],
    ) -> Dict[int, OrdinalEncoder] | None:
        categorical_encoders = {}
        for fold in range(self.num_folds):
            train_fold = train_df[train_df.kfold != fold].reset_index(drop=True)
            valid_fold = train_df[train_df.kfold == fold].reset_index(drop=True)
            if test_df is not None:
                test_fold = test_df.copy(deep=True)

            if len(categorical_features) > 0:
                ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

                train_fold_values = train_fold[categorical_features].values
                train_fold[categorical_features] = ordinal_encoder.fit_transform(train_fold_values)

                valid_fold_values = valid_fold[categorical_features].values
                valid_fold[categorical_features] = ordinal_encoder.transform(valid_fold_values)

                if test_df is not None:
                    test_fold_values = test_fold[categorical_features].values
                    test_fold[categorical_features] = ordinal_encoder.transform(test_fold_values)

                categorical_encoders[fold] = ordinal_encoder

            train_fold_path = fs.get_model_fold_path(self.model_id, None, "train", fold)
            train_fold.to_feather(train_fold_path)

            valid_fold_path = fs.get_model_fold_path(self.model_id, None, "valid", fold)
            valid_fold.to_feather(valid_fold_path)

            if test_df is not None:
                test_fold_path = fs.get_model_fold_path(self.model_id, None, "test", fold)
                test_fold.to_feather(test_fold_path)

        return categorical_encoders

    def prepare(self):
        self._process_data()

    def get_learner(self):
        learner = Learner(
            problem_type=self.model_config.problem_type,
            model_id=self.model_id,
            features=self.features,
            targets=self.targets,
            num_folds=self.num_folds,
            num_trials=self.num_trials,
            has_tests=self.has_tests,
            time_limit=self.time_limit,
            use_gpu=self.use_gpu,
            seed=self.seed,
            idx=self.idx,
        )

        return learner


def _get_processed_df_from_csv(
        model_id: str,
        dataset_type: fs.DatasetType,
        idx: Optional[str],
) -> pd.DataFrame | None:
    dataset_path = fs.get_dataset_path(model_id, dataset_type)
    if not dataset_path.is_file():
        return None

    df = pd.read_csv(dataset_path)
    df = reduce_memory_usage(df)
    df = _inject_idx(df, idx)

    return df


def _inject_idx(df: pd.DataFrame, idx: Optional[int]) -> pd.DataFrame:
    if idx is None:
        return df

    if idx not in df.columns:
        df[idx] = np.arange(len(df))

    return df


def _create_folds(
        df: pd.DataFrame,
        targets: List[str],
        num_folds: int,
        problem: ProblemType,
        seed: int,
) -> pd.DataFrame:
    df["kfold"] = -1
    if problem in (ProblemType.binary_classification, ProblemType.multi_class_classification):
        y = df[targets].values
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indices, "kfold"] = fold
    elif problem == ProblemType.single_column_regression:
        y = df[targets].values
        num_bins = int(np.floor(1 + np.log2(len(df))))
        if num_bins > 10:
            num_bins = 10
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        df["bins"] = pd.cut(df[targets].values.ravel(), bins=num_bins, labels=False)
        for fold, (_, valid_indices) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indices, "kfold"] = fold
        df = df.drop("bins", axis=1)
    elif problem == ProblemType.multi_column_regression:
        y = df[targets].values
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indices, "kfold"] = fold
    elif problem == ProblemType.multi_label_classification:  # TODO: use `iterstrat`
        y = df[targets].values
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indices, "kfold"] = fold
    else:
        raise Exception("Problem type not supported")

    return df


def _save_model_config(model: Any, model_id: str, model_version: Optional[fs.DatasetVersion]):
    filename = f"{model_id}.config"
    logger.info(f"Saving model config: {filename}")
    fs.save_object(model, model_id, filename, model_version)


def _save_categorical_encoders(model: Any, model_id: str, model_version: Optional[fs.DatasetVersion]):
    filename = f"{model_id}.categorical_encoders"
    logger.info(f"Saving categorical encoders: {filename}")
    fs.save_object(model, model_id, filename, model_version)


def _save_target_encoder(model: Any, model_id: str, model_version: Optional[fs.DatasetVersion]):
    filename = f"{model_id}.target_encoder"
    logger.info(f"Saving target encoder: {filename}")
    fs.save_object(model, model_id, filename, model_version)
