from datetime import datetime
import os
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split

date_fmt = "%Y-%m-%d-%H-%M-%S"
model_dates = [
    datetime.strptime(x.split(".model")[0], date_fmt)
    for x in os.listdir("app/models/")
    if x.endswith(".model")
]

for model_date in tqdm(model_dates):
    model_path = "app/models/{}.model".format(model_date.strftime(date_fmt))
    model = joblib.load(model_path)
    joblib.dump(model, model_path + ".compressed", compress=2)
