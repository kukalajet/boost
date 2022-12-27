import dataclasses
from typing import Optional, List
from pandas import pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import logger
from problem import determine_problem_type
from utils import inject_idx, create_folds
from problem import ProblemType


@dataclasses
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
        train_df = pd.read_csv(self.train_filename)
        # TODO: use `reduce_memory_usage` here

        problem = determine_problem_type(self.targets, train_df, self.task)
        train_df = inject_idx(train_df)

        if self.test_filename is not None:
            test_df = pd.read_csv(self.test_filename)
            # TODO: use `reduce_memory_usage` here
            test_df = inject_idx(train_df)

        train_df = create_folds(train_df, problem)
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

        if len(self.categorical_features) > 0:
            logger.info("Encoding categorical features")

    def train(self):
        return

    def predict(self):
        return
