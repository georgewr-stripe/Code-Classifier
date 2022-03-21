import os
import joblib
from typing import Union, Iterable
from datetime import datetime
from functools import cache

from model import preprocess  # required to load the model


class Classifier:
    def __init__(self) -> None:
        # Load the model
        date_fmt = "%Y-%m-%d-%H-%M-%S"
        latest_model_date = sorted(
            [
                datetime.strptime(x.split(".model.compressed")[0], date_fmt)
                for x in os.listdir("data/models/")
                if x.endswith(".model")
            ]
        )[-1]
        self.model_path = "data/models/{}.model.compressed".format(
            latest_model_date.strftime(date_fmt)
        )
        self.model = joblib.load(self.model_path)

    def predict(self, input: Union[tuple, str]) -> tuple:

        if isinstance(input, str):
            input = input

        predictions = cache(self.model.predict)(input)

        return tuple(predictions)
