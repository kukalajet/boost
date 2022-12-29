import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from boost.utils import inject_idx, create_folds
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

        problem = get_problem_type(self.targets, train_df, self.task)

        train_df = create_folds(train_df, self.targets, self.num_folds, problem, self.seed)
        ignore_columns = [self.idx, "kfold"] + self.targets

        if self.features is None:
            self.features = list(train_df.columns)
            self.features = [x for x in self.features if x not in ignore_columns]

        if problem in [ProblemType.binary_classification, ProblemType.multi_class_classification]:
            target_encoder = LabelEncoder()
            target_values = train_df[self.targets].values
            target_encoder.fit(target_values.reshape(-1))
            train_df.loc[:, self.targets] = target_encoder.transform(
                target_values.reshape(-1))
        else:
            target_encoder = None

        if self.categorical_features is None:
            categorical_features = []
            for feature in self.features:
                if train_df[feature].dtype == "object":
                    categorical_features.append(feature)
        else:
            categorical_features = self.categorical_features

        logger.info(f"Found {len(categorical_features)} categorical features.")

        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")

        categorical_encoders = {}
        for fold in range(self.num_folds):
            fold_train = train_df[train_df.kfold != fold].reset_index(drop=True)
            fold_valid = train_df[train_df.kfold == fold].reset_index(drop=True)

            if self.test_filename is not None:
                test_fold = test_df.copy(deep=True)

            if len(categorical_features) > 0:
                ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                fold_train[categorical_features] = ordinal_encoder.fit_transform(
                    fold_train[categorical_features].values)
                fold_valid[categorical_features] = ordinal_encoder.transform(
                    fold_valid[categorical_features].values)
                if self.test_filename is not None:
                    test_fold[categorical_features] = ordinal_encoder.transform(
                        test_fold[categorical_features].values)
                categorical_encoders[fold] = ordinal_encoder

            train_fold_path = f"../{self.output}/train_fold_{fold}.feather"
            valid_fold_path = f"../{self.output}/valid_fold_{fold}.feather"

            fold_train.to_feather(train_fold_path)
            fold_valid.to_feather(valid_fold_path)

            if self.test_filename is not None:
                test_fold.to_feather(os.path.join("..", self.output, f"test_fold_{fold}.feather"))

        self.model_config = ModelConfig(
            idx=self.idx,
            features=self.features,
            categorical_features=categorical_features,
            train_filename=self.train_filename,
            test_filename=self.test_filename,
            output=self.output,
            problem=problem,
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

        # save encoders
        logger.info("Saving encoders")

        categorical_encoders_path = f"../{self.output}/axgb.categorical_encoders"
        target_encoder_path = f"../{self.output}/axgb.target_encoder"

        joblib.dump(categorical_encoders, categorical_encoders_path)
        joblib.dump(target_encoder, target_encoder_path)

    def train(self):
        self._process_data()
        params = train_model(self.model_config)
        logger.info("Training complete")
        self.predict(params)

    def predict(self, params: Dict[str, Any]):
        logger.info("Creating OOF and test predictions")
        predict_model(self.model_config, params)


def get_processed_df_from_csv(filename: Optional[str], idx: Optional[str]) -> pd.DataFrame | None:
    if filename is None:
        return None

    path_to_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', filename))
    df = pd.read_csv(path_to_csv)
    # TODO: use `reduce_memory_usage` here
    df = inject_idx(df, idx)

    return df
