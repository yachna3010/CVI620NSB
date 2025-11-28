# q1_infer_cats_dogs.py
# Use the saved Cat vs Dog model on Q1/test or custom images.

import os
import sys
from typing import List

import numpy as np
from PIL import Image
import joblib

IMAGE_SIZE_DEFAULT = (64, 64)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_model_bundle(base_dir: str):
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "cat_dog_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run q1_train_cats_dogs.py first to train and save the model."
        )
    bundle = joblib.load(model_path)
    return bundle


def preprocess_image(path: str, image_size) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def collect_image_paths(args: List[str]) -> List[str]:
    image_paths: List[str] = []
    for p in args:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(VALID_EXTENSIONS):
                        image_paths.append(os.path.join(root, f))
        elif os.path.isfile(p):
            if p.lower().endswith(VALID_EXTENSIONS):
                image_paths.append(p)
        else:
            print(f"[WARN] Path does not exist and will be skipped: {p}")
    return image_paths


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundle = load_model_bundle(base_dir)

    model = bundle["model"]
    class_names = bundle["class_names"]
    image_size = bundle.get("image_size", IMAGE_SIZE_DEFAULT)

    # If no arguments are given, default to using all images in Q1/test
    if len(sys.argv) <= 1:
        default_test_dir = os.path.join(base_dir, "test")
        print(
            "No image paths provided. "
            f"Using all images found in: {default_test_dir}"
        )
        args = [default_test_dir]
    else:
        args = sys.argv[1:]

    image_paths = collect_image_paths(args)

    if not image_paths:
        print("No images found to classify. Exiting.")
        return

    print(f"Found {len(image_paths)} image(s) to classify.\n")

    features = []
    valid_paths = []
    for path in image_paths:
        try:
            feat = preprocess_image(path, image_size)
            features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    if not features:
        print("No valid images to classify. Exiting.")
        return

    X = np.vstack(features)
    preds = model.predict(X)

    print("Predictions:")
    for path, label_idx in zip(valid_paths, preds):
        label_name = class_names[label_idx]
        print(f"{path} -> {label_name}")


if __name__ == "__main__":
    main()
