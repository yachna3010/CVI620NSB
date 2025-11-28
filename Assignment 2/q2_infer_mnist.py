# q2_infer_mnist.py
# Use the saved MNIST model to print predictions for some test digits.

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import joblib


def load_model_bundle(base_dir: str):
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "mnist_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run q2_train_mnist.py first to train and save the model."
        )
    return joblib.load(model_path)


def predict_from_test_set(
    bundle,
    test_csv_path: str,
    indices: List[int] = None,
):
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    # No header, first column is label, rest are pixels
    test_df = pd.read_csv(test_csv_path, header=None)

    y_test = test_df.iloc[:, 0].values.astype(np.int64)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0

    model = bundle["model"]
    class_names = bundle.get("class_names", [str(i) for i in range(10)])

    if not indices:
        indices = list(range(10))  # show first 10 digits by default

    print("Sample predictions from mnist_test.csv:\n")
    for idx in indices:
        if idx < 0 or idx >= len(X_test):
            print(f"[WARN] Index {idx} is out of range for test set.")
            continue

        x = X_test[idx].reshape(1, -1)
        y_true = y_test[idx]
        y_pred = model.predict(x)[0]

        print(
            f"Index {idx:5d}: True = {class_names[y_true]}, "
            f"Predicted = {class_names[y_pred]}"
        )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv_path = os.path.join(base_dir, "mnist_test.csv")
    bundle = load_model_bundle(base_dir)

    # If indices are passed on command line, use them; else default 0–9
    if len(sys.argv) > 1:
        try:
            idx_list = [int(arg) for arg in sys.argv[1:]]
        except ValueError:
            print("All indices must be integers. Example: python q2_infer_mnist.py 0 1 2")
            return
    else:
        idx_list = list(range(10))

    predict_from_test_set(bundle, test_csv_path, idx_list)


if __name__ == "__main__":
    main()
