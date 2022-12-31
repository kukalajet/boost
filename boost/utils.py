import joblib
import os
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from boost.problem import ProblemType


def inject_idx(df: pd.DataFrame, idx: Optional[int]) -> pd.DataFrame:
    if idx is None:
        return df

    if idx not in df.columns:
        df[idx] = np.arange(len(df))

    return df


def create_folds(
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


def mean_dict_values(values: List[Dict[str, int]]) -> Dict[str, Any]:
    mean_dict = {}
    for key in values[0].keys():
        mean_dict[key] = sum(d[key] for d in values) / len(values)

    return mean_dict


def get_processed_df_from_csv(filename: Optional[str], idx: Optional[str]) -> pd.DataFrame | None:
    if filename is None:
        return None

    path_to_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', filename))
    df = pd.read_csv(path_to_csv)
    # TODO: use `reduce_memory_usage` here
    df = inject_idx(df, idx)

    return df


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
