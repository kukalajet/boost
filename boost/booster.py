from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from boost.utils import create_folds, get_fold_path, get_processed_df_from_csv, persist_object
from boost.problem import ProblemType, get_problem_type
from boost.model import ModelConfig, train_model, predict_model
from boost.logger import logger


@dataclass
class Booster:
    train_filename: str
    output: str
    test_filename: Optional[str] = None
    task: Optional[str] = None
    idx: Optional[str] = "id"
    targets: Optional[List[str]] = None
    features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    use_gpu: Optional[bool] = False
    num_folds: Optional[int] = 5
    seed: Optional[int] = 42
    num_trials: Optional[int] = 1000
    time_limit: Optional[int] = None
    fast: Optional[bool] = False

    def __post_init__(self):
        if self.targets is None:
            logger.warning("No target columns specified. Will default to `target`.")
            self.targets = ['target']
        if self.idx is None:
            logger.warning("No id column specified. Will default to `id`.")
            self.idx = "id"

    def _process_data(self):
        train_df = get_processed_df_from_csv(self.train_filename, self.idx)
        test_df = get_processed_df_from_csv(self.test_filename, self.idx)

        problem_type = get_problem_type(self.targets, train_df, self.task)
        train_df = self._create_folds(train_df, problem_type)
        target_encoder, train_df = self._get_target_encoder(train_df, problem_type)

        features = self._get_features(train_df)
        categorical_features = self._get_categorical_features(train_df, features)

        logger.info(f"Found {len(categorical_features)} categorical features.")
        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")

        categorical_encoders = self._get_categorical_encoders(train_df, test_df, categorical_features)

        self.model_config = ModelConfig(
            idx=self.idx,
            features=features,
            categorical_features=categorical_features,
            train_filename=self.train_filename,
            test_filename=self.test_filename,
            output=self.output,
            problem_type=problem_type,
            targets=self.targets,
            use_gpu=self.use_gpu,
            num_folds=self.num_folds,
            seed=self.seed,
            num_trials=self.num_trials,
            time_limit=self.time_limit,
            fast=self.fast,
        )

        logger.info(f"Model config: {self.model_config}")
        logger.info("Saving model config")
        logger.info("Saving encoders")

        persist_object(categorical_encoders, self.output, "axgb.categorical_encoders")
        persist_object(target_encoder, self.output, "axgb.target_encoder")

    def _create_folds(self, train_df: pd.DataFrame, problem: ProblemType) -> pd.DataFrame:
        if "kfold" in train_df.columns:
            self.num_folds = len(np.unique(train_df["kfold"]))
            logger.info("Using `kfold` for folds from training data")
            return train_df

        train_df = create_folds(train_df, self.targets, self.num_folds, problem, self.seed)
        return train_df

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

            train_fold_path = get_fold_path(self.output, fold, "train")
            train_fold.to_feather(train_fold_path)

            valid_fold_path = get_fold_path(self.output, fold, "valid")
            valid_fold.to_feather(valid_fold_path)

            if test_df is not None:
                test_fold_path = get_fold_path(self.output, fold, "test")
                test_fold.to_feather(test_fold_path)

        return categorical_encoders

    def train(self):
        self._process_data()
        params = train_model(self.model_config)
        logger.info("Training complete")
        self.predict(params)

    def predict(self, params: Dict[str, Any]):
        logger.info("Creating OOF and test predictions")
        predict_model(self.model_config, params)
