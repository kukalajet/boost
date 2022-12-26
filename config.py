from typing import List, Optional
from pydantic import BaseModel
from problem import ProblemType


class ModelConfig(BaseModel):
    train_filename: str
    test_filename: Optional[str] = None
    idx: str
    targets: List[str]
    problem: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool
