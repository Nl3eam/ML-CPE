import json
import os

import numpy as np
from tensorflow import keras

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from vgg_model import train_model, predict_model, VGG_1D, VGG_SMALL
from evaluate import evaluate_model, plot_history

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "ramen-ratings.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TARGET = "Style"        # "Style" (Pack/Bowl/Cup/Tray) or "Rating" (Low/Medium/High)
MIN_PER_CLASS = 30      # drop classes with fewer rows than this
MAX_LEN = 96            # characters per sample (99% of rows are <= 84)
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EPOCHS = 70
BATCH_SIZE = 32

# VGG_1D = 7 conv layers (2-2-3). VGG_SMALL = lighter.
BLOCKS = VGG_1D
SEED = 42               # fixed for reproducible runs


def main():

    print("--" * 30)
    print(f"VGG-style 1D DCNN on Ramen Ratings: predict {TARGET}")
    print("--" * 30)

    keras.utils.set_random_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    texts, labels, classes = load_data(DATA_PATH, TARGET, MIN_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total rows : {len(texts)}")
    print(f"Classes    : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Encoding text to character ids...")

    X = to_features(texts, MAX_LEN)
    y = labels

    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE, BLOCKS
    )

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()