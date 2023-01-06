from boost.data_loader import DataLoader
from boost.predictor import Predictor

# from boost.booster import Booster

train_filename = "train.csv"
model_id = "binary_classification"
test_filename = None
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
# fast = False
#
data_loader = DataLoader(
    model_id=model_id,
    train_filename=train_filename,
    test_filename=test_filename,
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
#
predictor = Predictor(model_id=model_id)

# test = Booster(
#     train_filename=train_filename,
#     output=output,
#     test_filename=test_filename,
#     task=task,
#     idx=idx,
#     targets=targets,
#     features=features,
#     categorical_features=categorical_features,
#     use_gpu=use_gpu,
#     num_folds=num_folds,
#     seed=seed,
#     num_trials=num_trials,
#     time_limit=time_limit,
#     fast=fast,
# )
#
# test.train()

#####

# predictor = Predictor(model_id="output")

# value = '''{
#   "age": "44",
#   "workclass": "Private",
#   "fnlwgt": "121874",
#   "education": "Some-college",
#   "education.num": "10",
#   "marital.status": "Divorced",
#   "occupation": "Sales",
#   "relationship": "Unmarried",
#   "race": "White",
#   "sex": "Male",
#   "capital.gain": "0",
#   "capital.loss": "0",
#   "hours.per.week": "55",
#   "native.country": "United-States"
# }'''
# value = '''{
#   "age": "36",
#   "workclass": "Private",
#   "fnlwgt": "169426",
#   "education": "HS-grad",
#   "education.num": "9",
#   "marital.status": "Married-civ-spouse",
#   "occupation": "Machine-op-inspct",
#   "relationship": "Husband",
#   "race": "White",
#   "sex": "Male",
#   "capital.gain": "7298",
#   "capital.loss": "0",
#   "hours.per.week": "40",
#   "native.country": "United-States"
# }'''
# value = '''{
#   "age": "52",
#   "workclass": "Self-emp-inc",
#   "fnlwgt": "334273",
#   "education": "Doctorate",
#   "education.num": "16",
#   "marital.status": "Married-civ-spouse",
#   "occupation": "Prof-specialty",
#   "relationship": "Husband",
#   "race": "White",
#   "sex": "Male",
#   "capital.gain": "99999",
#   "capital.loss": "0",
#   "hours.per.week": "65",
#   "native.country": "United-States"
# }'''
# value = '''{
#   "age": "24",
#   "workclass": "Private",
#   "fnlwgt": "359828",
#   "education": "Bachelors",
#   "education.num": "13",
#   "marital.status": "Married-civ-spouse",
#   "occupation": "Prof-specialty",
#   "relationship": "Husband",
#   "race": "White",
#   "sex": "Male",
#   "capital.gain": "0",
#   "capital.loss": "0",
#   "hours.per.week": "44",
#   "native.country": "United-States"
# }'''
# value = '''{
#   "age": "52",
#   "workclass": "Private",
#   "fnlwgt": "97411",
#   "education": "5th-6th",
#   "education.num": "3",
#   "marital.status": "Married-civ-spouse",
#   "occupation": "Machine-op-inspct",
#   "relationship": "Husband",
#   "race": "Asian-Pac-Islander",
#   "sex": "Male",
#   "capital.gain": "0",
#   "capital.loss": "0",
#   "hours.per.week": "40",
#   "native.country": "Laos"
# }'''
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
#
res = predictor.predict(value)
print(res)
