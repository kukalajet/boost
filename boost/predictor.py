import os
import json
from typing import Any, List, Dict

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
from pydantic import create_model

from boost.model import ModelConfig
from boost.problem import ProblemType
from boost.utils import load_persisted_object
from boost.model import get_model_and_hyperparameters
from boost.logger import logger


class Predictor:
    def __init__(self, model_folder: str, idx: str = None):
        self.model_folder = model_folder

        self.model_config = _load_model_config(self.model_folder)
        self.target_encoder = _load_target_encoder(self.model_folder)
        self.categorical_encoders = _load_categorical_encoders(self.model_folder)
        self.models = _load_models(self.model_folder, self.model_config.num_folds)
        _, self.predict_probabilities, _, _ = get_model_and_hyperparameters(self.model_config.problem_type)

        self.idx = idx
        if self.idx is None:
            logger.warning("No id column specified. Will default to `id`.")
            self.idx = "id"

    def get_prediction_schema(self):
        schema = {"schema": {}}

        for categorical_feature in self.model_config.categorical_features:
            schema["schema"][categorical_feature] = "str"

        for feature in self.model_config.features:
            if feature not in self.model_config.categorical_features:
                schema["schema"][feature] = 10.0

        return create_model("Schema", **schema["schema"])

    def _predict_from_df(self, df: pd.DataFrame):
        categorical_features = self.model_config.categorical_features
        test_ids = df[self.model_config.idx].values

        final_predictions = []
        for fold in range(self.model_config.num_folds):
            fold_test = df.copy(deep=True)
            if len(categorical_features) > 0:
                categorical_encoder = self.categorical_encoders[fold]
                fold_test[categorical_features] = categorical_encoder.transform(fold_test[categorical_features].values)
            test_features = fold_test[self.model_config.features].copy()

            for column in test_features.columns:
                if test_features[column].dtype == "object":
                    test_features.loc[:, column] = test_features[column].astype(np.int64)

            if self.model_config.problem_type in (
                    ProblemType.multi_column_regression, ProblemType.multi_label_classification):
                predictions = []
                for index in range(len(self.models[fold])):
                    if self.model_config.problem_type == ProblemType.multi_column_regression:
                        prediction = self.models[fold][index].predict(test_features)
                    else:
                        prediction = self.models[fold][index].predict_proba(test_features)[:, 1]
                    predictions.append(prediction)
            else:
                if self.predict_probabilities:
                    predictions = self.models[fold].predict_proba(test_features)
                else:
                    predictions = self.models[fold].predict(test_features)

            final_predictions.append(predictions)

        final_predictions = np.mean(final_predictions, axis=0)
        if self.target_encoder is None:
            final_predictions = pd.DataFrame(final_predictions, columns=self.model_config.targets)
        else:
            final_predictions = pd.DataFrame(final_predictions, columns=list(self.target_encoder.classes_))

        final_predictions.insert(loc=0, column=self.model_config.idx, value=test_ids)
        return final_predictions

    def predict(self, sample: str = None):
        sample = json.loads(sample)
        sample_df = pd.DataFrame.from_dict(sample, orient="index").T
        sample_df[self.idx] = 0
        predictions = self._predict_from_df(sample_df)
        predictions = predictions.to_dict(orient="records")[0]
        return predictions


def _load_model_config(model_folder: str) -> ModelConfig:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_folder, "axgb.config"))
    model_config = load_persisted_object(path)
    return model_config


def _load_target_encoder(model_folder: str) -> LabelEncoder | OrdinalEncoder:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_folder, "axgb.target_encoder"))
    target_encoder = load_persisted_object(path)
    return target_encoder


def _load_categorical_encoders(model_folder: str) -> Dict[int, OrdinalEncoder] | None:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_folder, "axgb.categorical_encoders"))
    categorical_encoders = load_persisted_object(path)
    return categorical_encoders


# How cool would it be if we had an actual type here? :)
# TODO: debug and find out what type models are.
def _load_models(model_folder: str, num_folds: int) -> List[Any]:
    models = []
    for fold in range(num_folds):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_folder, f"axgb_model.{fold}"))
        model = load_persisted_object(path)
        models.append(model)

    return models
