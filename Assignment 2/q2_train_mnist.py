# q2_train_mnist.py
# Train several models on MNIST CSV and save the best one.

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def load_mnist(
    train_csv: str, test_csv: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST train and test CSV files.

    Assumes:
      - There is NO header row.
      - Column 0 = label (digit 0–9).
      - Columns 1..end = pixel values.
    """

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # header=None -> pandas will not treat first row as header
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    # First column is label, the rest are pixels
    y_train_full = train_df.iloc[:, 0].values.astype(np.int64)
    X_train_full = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0

    y_test = test_df.iloc[:, 0].values.astype(np.int64)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0

    return X_train_full, y_train_full, X_test, y_test


def train_and_select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """Train multiple models and return the best one based on validation accuracy."""
    models = {
        # Multinomial logistic regression
        "logreg": LogisticRegression(
            max_iter=250,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            C=1.5,
        ),
        # Simple neural net
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=30,  # increase if you want even higher accuracy
            random_state=42,
        ),
        # Random forest
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_acc = 0.0

    for name, clf in models.items():
        print(f"\n=== Training model: {name} ===")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{name} validation accuracy: {acc:.4f}")
        print("Classification report (validation):")
        print(classification_report(y_val, y_pred))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = clf

    print(f"\nBest validation model: {best_name} with accuracy {best_acc:.4f}")
    return best_name, best_model, best_acc


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "mnist_train.csv")
    test_csv = os.path.join(base_dir, "mnist_test.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading MNIST data from:\n  {train_csv}\n  {test_csv}")
    X_train_full, y_train_full, X_test, y_test = load_mnist(train_csv, test_csv)
    print(f"Training samples: {X_train_full.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train_full.shape[1]}")

    # Split off validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42,
    )
    print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    best_name, best_model, best_val_acc = train_and_select_model(
        X_train, y_train, X_val, y_val
    )

    # Evaluate on official test set
    print("\n=== Evaluation on official test set ===")
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, y_test_pred))

    if test_acc < 0.90:
        print(
            "\n[NOTE] Test accuracy is below 90%. "
            "You can increase max_iter for the MLP or logistic regression "
            "to improve performance."
        )

    # Save everything
    bundle = {
        "model": best_model,
        "best_model_name": best_name,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "class_names": [str(i) for i in range(10)],
    }

    model_path = os.path.join(models_dir, "mnist_best_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"\nSaved best MNIST model bundle to: {model_path}")


if __name__ == "__main__":
    main()
