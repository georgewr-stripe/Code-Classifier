from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from api._model import model

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# Loading Data
DATA_PATH = r"data/train/data.feather"
MODEL_PATH = r"api/models/{}.model.compressed".format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
)

with tqdm(total=100) as pbar:

    pbar.set_description("Loading Data...")

    data = pd.read_feather(DATA_PATH)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data = data.head(5000)

    # Train/Test split
    pbar.update(20)
    pbar.set_description("Splitting Test/Train...")
    X, y = data.content, data.language
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fitting
    pbar.update(30)
    pbar.set_description("Training Model ({} samples)...".format(X_train.shape[0]))

    space = [
        Categorical(["tanh", "relu"], name="activation"),
        Integer(1, 4, name="n_hidden_layer"),
        Integer(10, 1000, name="n_neurons_per_layer"),
    ]

    @use_named_args(space)
    def objective(**params):
        n_neurons = params["n_neurons_per_layer"]
        n_layers = params["n_hidden_layer"]

        # create the hidden layers as a tuple with length n_layers and n_neurons per layer
        params["hidden_layer_sizes"] = (n_neurons,) * n_layers

        # the parameters are deleted to avoid an error from the MLPRegressor
        params.pop("n_neurons_per_layer")
        params.pop("n_hidden_layer")

        # Prefix as we are using a pipeline
        model.set_params(**{"clf__" + k: v for k, v in params.items()})

        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        return -np.mean(
            cross_val_score(
                model, X, y, cv=5, n_jobs=-4, scoring="neg_mean_absolute_error"
            )
        )

    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
    print(res_gp)

    # model.fit(X_train, y_train)

    # Save Model
    pbar.update(20)
    pbar.set_description("Saving Model...")
    # joblib.dump(model, MODEL_PATH, compress=9)

    # Evaluation
    pbar.update(20)
    pbar.set_description("Evaluating Model...")
    # print(f"Accuracy: {(model.score(X_test, y_test) * 100).round(2)}%")
    pbar.update(10)
