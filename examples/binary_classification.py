from boost.data_loader import DataLoader
from boost.predictor import Predictor

model_id = "binary_classification"
task = None
idx = None
targets = ["income"]
features = None
categorical_features = None
use_gpu = False
num_folds = 5
seed = 42
num_trials = 100
time_limit = 360

data_loader = DataLoader(
    model_id=model_id,
    task=task,
    idx=idx,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
)
data_loader.prepare()
learner = data_loader.get_learner()
learner.train()

predictor = Predictor(model_id=model_id)

value = '''{
  "age": "44",
  "workclass": "Private",
  "fnlwgt": "277647",
  "education": "HS-grad",
  "education.num": "9",
  "marital.status": "Married-civ-spouse",
  "occupation": "Sales",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital.gain": "0",
  "capital.loss": "1902",
  "hours.per.week": "40",
  "native.country": "United-States"
}'''

prediction = predictor.predict(value)
print(f"Prediction: {prediction}")
