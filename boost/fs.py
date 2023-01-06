from typing import Literal, Any

from pathlib import Path
import joblib

CURRENT_DIRECTORY = Path(__file__).parent
ROOT_DIRECTORY = CURRENT_DIRECTORY.parent
MODELS_PATH = ROOT_DIRECTORY / Path("models")
CONFIG_PATH = ROOT_DIRECTORY / Path("config")
DATASETS_FOLDER = Path("datasets")

PredictionType = Literal["valid", "test"]
ModelVersion = str | Literal["latest"]
DatasetType = Literal["train", "valid", "test"]
DatasetVersion = str | Literal["latest"]
ModelFoldType = Literal["train", "valid", "test"]


def get_model_path(model_id: str):
    """Returns the path to the model given the `model_id` and the `model_version`"""
    return MODELS_PATH / model_id


def _get_latest_model_version(model_id: str):
    """Returns the latest version stored of a model given the `model_id`"""
    # Default to `0` at the moment.
    # TODO: Needs to be implemented once I have figured out the best folder structure.
    return "0"


def get_dataset_path(model_id: str, dataset_type: DatasetType) -> Path:
    """Returns the path to the dataset given the `model_id` and the `dataset_type`"""
    model_path = get_model_path(model_id)
    return model_path / DATASETS_FOLDER / f"{dataset_type}.csv"


def get_optuna_database_path() -> Path:
    """Returns the path to the optuna database"""
    return CONFIG_PATH / "hyperparams.db"


def get_predictions_path(
        model_id: str,
        model_version: ModelVersion | None,
        prediction_type: PredictionType | None,
) -> Path:
    """Returns the path to the predictions given the `model_id`"""
    model_version = model_version if model_version is not None else _get_latest_model_version(model_id)
    prediction_type = prediction_type if prediction_type is not None else "valid"
    return MODELS_PATH / model_id / model_version / f"oof-{prediction_type}-predictions.csv"


def get_model_fold_path(
        model_id: str,
        model_version: ModelVersion | None,
        model_fold_type: ModelFoldType,
        fold: int,
) -> Path:
    model_version = model_version if model_version is not None else _get_latest_model_version(model_id)
    parent_folder = MODELS_PATH / model_id / model_version
    if not parent_folder.is_file():
        parent_folder.mkdir(parents=True, exist_ok=True)
    return parent_folder / f"{model_fold_type}_fold_{fold}.feather"


def load_object(model_id: str, model_version: ModelVersion | None, filename: str) -> Any:
    """Reconstruct a Python object from a file given the `model_id` and the `filename`"""
    model_version = model_version if model_version is not None else _get_latest_model_version(model_id)
    path = MODELS_PATH / model_id / model_version / filename
    value = joblib.load(path)
    return value


def save_object(value: Any, model_id, filename: str, model_version: ModelVersion | None):
    """Persist an arbitrary Python object into one file given the object (`value`), the `model_id` and the `filename`."""
    model_version = model_version if model_version is not None else "0"
    parent_folder = MODELS_PATH / model_id / model_version
    if not parent_folder.is_file():
        parent_folder.mkdir(parents=True, exist_ok=True)
    path = parent_folder / filename
    joblib.dump(value, path)
