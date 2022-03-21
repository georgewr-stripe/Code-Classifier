from datetime import datetime
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from app.model import model

# Loading Data
DATA_PATH = r"data/train/data.feather"
MODEL_PATH = r"app/models/{}.model.compressed".format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
)

with tqdm(total=100) as pbar:

    pbar.set_description("Loading Data...")

    data = pd.read_feather(DATA_PATH)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # Train/Test split
    pbar.update(20)
    pbar.set_description("Splitting Test/Train...")
    X, y = data.content, data.language
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fitting
    pbar.update(30)
    pbar.set_description("Training Model ({} samples)...".format(X_train.shape[0]))
    model.fit(X_train, y_train)

    # Save Model
    pbar.update(20)
    pbar.set_description("Saving Model...")
    joblib.dump(model, MODEL_PATH, compress=1)

    # Evaluation
    print(f"Accuracy: {model.score(X_test, y_test)}")
