import imp
from classify import predict
import time
from tqdm import tqdm

tests = [
    ("javascript", "const i = new RegExp(); console.log(i)"),
    ("python", "def foo(args, **kwargs): return i"),
]

if __name__ == "__main__":

    codes = tuple([x[1] for x in tests])
    languages = tuple([x[0] for x in tests])
    predictions = []

    with tqdm(total=len(tests) + 1) as pbar:
        for i, code in enumerate(codes):
            pbar.set_description(
                "Running Individual Test {}/{}...".format(i + 1, len(tests))
            )
            predictions.append(predict(code))
            pbar.update(1)

        pbar.set_description("Running Batch Test...")
        predictions = predict(codes)
        for i, pred in enumerate(predictions):
            print("Testing for {}, Prediction: {}".format(tests[i][0], pred))
        pbar.update(1)
