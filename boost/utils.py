import joblib
import os
from typing import Any, Literal


def get_fold_path(output: str, fold: int, set_type: Literal["train", "valid", "test"]) -> str:
    filename = f"{set_type}_fold_{fold}.feather"
    fold_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output, filename))
    return fold_path


def persist_object(value: Any, relative_path: str, filename: str):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', relative_path, filename))
    joblib.dump(value, path)


def load_persisted_object(path: str):
    value = joblib.load(path)
    return value
