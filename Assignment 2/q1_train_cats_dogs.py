# q1_train_cats_dogs.py
# Train several models on Cat vs Dog and save the best one.

import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

IMAGE_SIZE = (64, 64)        # all images resized to 64x64
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_image(path: str) -> np.ndarray:
    """Load a single image, convert to grayscale, resize and flatten."""
    img = Image.open(path).convert("L")          # grayscale
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def load_dataset(root_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images from a root directory with subfolders per class.
    Example structure:
        root_dir/
            Cat/
            Dog/
    Returns X, y and the list of class names indexed by label.
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    class_names: List[str] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    # Ensure consistent label order (alphabetical)
    for label, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_names.append(class_name)

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue
            file_path = os.path.join(class_dir, file_name)
            try:
                X.append(load_image(file_path))
                y.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {file_path}: {e}")

    if not X:
        raise RuntimeError(f"No images found in {root_dir}")

    X_arr = np.vstack(X)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr, class_names


def train_and_select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """
    Train multiple models and return the best one based on validation accuracy.
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=500,
            C=2.0,
            solver="lbfgs",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_acc = 0.0

    for name, clf in models.items():
        print(f"\n=== Training model: {name} ===")

        # Same pipeline for all models (scale features)
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{name} validation accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_val, y_pred))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipe

    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    return best_name, best_model, best_acc


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Training directory: {train_dir}")
    X, y, class_names = load_dataset(train_dir)
    print(f"Loaded {len(X)} samples from {len(class_names)} classes: {class_names}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    best_name, best_model, best_acc = train_and_select_model(
        X_train, y_train, X_val, y_val
    )

    # Optional: evaluate on Q1/test if present
    if os.path.isdir(test_dir):
        print("\n=== Extra evaluation on Q1/test ===")
        X_test, y_test, _ = load_dataset(test_dir)
        y_pred_test = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"Test accuracy on Q1/test: {test_acc:.4f}")
        print("Classification report on Q1/test:")
        print(classification_report(y_test, y_pred_test))

    # Save everything needed in a single file
    bundle = {
        "model": best_model,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "best_model_name": best_name,
        "best_val_accuracy": best_acc,
    }

    model_path = os.path.join(models_dir, "cat_dog_best_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"\nSaved best model bundle to: {model_path}")


if __name__ == "__main__":
    main()
