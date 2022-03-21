import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


# Model params
token_pattern = r"""(\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""


def preprocess(x):
    return (
        pd.Series(x)
        .replace(r"\b([A-Za-z])\1+\b", "", regex=True)
        .replace(r"\b[A-Za-z]\b", "", regex=True)
    )


# Pipe steps
transformer = FunctionTransformer(preprocess)
vectorizer = TfidfVectorizer(token_pattern=token_pattern, max_features=3000)
# clf = RandomForestClassifier(n_jobs=4)
clf = MLPClassifier(
    solver="adam", alpha=1e-5, hidden_layer_sizes=(70, 50), random_state=1
)

model = Pipeline(
    [("preprocessing", transformer), ("vectorizer", vectorizer), ("clf", clf)],
)

# Setting best params
# best_params = {
#     "clf__criterion": "gini",
#     "clf__max_features": "sqrt",
#     "clf__min_samples_split": 3,
#     "clf__n_estimators": 300,
# }

# model.set_params(**best_params)
