from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from boost.problem import ProblemType


def inject_idx(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if idx not in df.columns:
        df[idx] = np.arange(len(df))

    return df


def create_folds(
        train_df: pd.DataFrame,
        targets: List[str],
        num_folds: int,
        problem: ProblemType,
        seed: int,
) -> pd.DataFrame:
    # TODO: What about this?
    # https://github.com/abhishekkrthakur/autoxgb/blob/334a0410466fbbd68d0f7c67acc2cb949b3d6fdc/src/autoxgb/autoxgb.py#L53

    train_df["kfold"] = -1
    if problem in (ProblemType.binary_classification, ProblemType.multi_class_classification):
        y = train_df[targets].values
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=train_df, y=y)):
            train_df.loc[valid_indices, "kfold"] = fold
    elif problem == ProblemType.single_column_regression:
        y = train_df[targets].values
        num_bins = int(np.floor(1 + np.log2(len(train_df))))
        if num_bins > 10:
            num_bins = 10
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        train_df["bins"] = pd.cut(train_df[targets].values.ravel(), bins=num_bins, labels=False)
        for fold, (_, valid_indices) in enumerate(kf.split(X=train_df, y=y)):
            train_df.loc[valid_indices, "kfold"] = fold
        train_df = train_df.drop("bins", axis=1)
    elif problem == ProblemType.multi_column_regression:
        y = train_df[targets].values
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=train_df, y=y)):
            train_df.loc[valid_indices, "kfold"] = fold
    elif problem == ProblemType.multi_label_classification:  # TODO: use `iterstrat`
        y = train_df[targets].values
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, valid_indices) in enumerate(kf.split(X=train_df, y=y)):
            train_df.loc[valid_indices, "kfold"] = fold
    else:
        raise Exception("Problem type not supported")

    return train_df


def dict_mean(dict_list) -> Dict[str, Any]:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)

    return mean_dict
