import os
from typing import Any, Literal

import joblib
import numpy as np

from boost.logger import logger


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


def reduce_memory_usage(df, verbose=True):
    # From: https://github.com/abhishekkrthakur/autoxgb/blob/414f6232b77f72713a293b83dd6a7f268153272b/src/autoxgb/utils.py#L20
    # NOTE: Original author of this function is unknown
    # if you know the *original author*, please let me know.
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df
