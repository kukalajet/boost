from boost.data_loader import DataLoader
from boost.predictor import Predictor

model_id = "multi_label_classification"
task = None
idx = "id"
targets = ["service_a", "service_b"]
features = None
categorical_features = None
use_gpu = False
num_folds = 5
seed = 42
num_trials = 100
time_limit = 360

# data_loader = DataLoader(
#     model_id=model_id,
#     task=task,
#     idx=idx,
#     targets=targets,
#     features=features,
#     categorical_features=categorical_features,
#     use_gpu=use_gpu,
#     num_folds=num_folds,
#     seed=seed,
#     num_trials=num_trials,
#     time_limit=time_limit
# )
# data_loader.prepare()
#
# learner = data_loader.get_learner()
# learner.train()

### value
value = '''{
    "id": "11193",
    "release": "a",
    "n_0047": "1",
    "n_0050": "1",
    "n_0052": "1",
    "n_0061": "1",
    "n_0067": "0.9285714285714286",
    "n_0075": "1",
    "n_0078": "0.8",
    "n_0091": "1",
    "n_0108": "0.8",
    "n_0109": "0.1875",
    "o_0176": "303",
    "o_0264": "7",
    "c_0466": "a",
    "c_0500": "a",
    "c_0638": "d",
    "c_0699": "b",
    "c_0738": "a",
    "c_0761": "c",
    "c_0770": "c",
    "c_0838": "a",
    "c_0870": "b",
    "c_0980": "c",
    "c_1145": "b",
    "c_1158": "g",
    "c_1189": "b",
    "c_1223": "c",
    "c_1227": "a",
    "c_1244": "d",
    "c_1259": "n"
}'''
predictor = Predictor(model_id="multi_label_classification")
prediction = predictor.predict(value)

print(f"Prediction: {prediction}")
