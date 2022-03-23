import json
from flask import Flask, request, jsonify

from typing import Union, List
from _classify import Classifier


app = Flask(__name__)
classifier = Classifier()


@app.route("/predict", methods=["POST"])
def predict():
    code = request.form.get("code")
    if not code:
        return jsonify(error="Please pass the code parameter"), 400

    predictions = classifier.predict(code)
    return jsonify(languages=predictions), 200
