import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import (
    build_vocabulary,
    texts_to_sequences,
    to_features,
    save_vocabulary,
)
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ramen-ratings.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MAX_LEN = 24           # words kept per review (like IMG_SIZE for images)
MAX_WORDS = 10000      # vocabulary cap
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = None   # None = use all rows
EPOCHS = 50
BATCH_SIZE = 8


def main():

    print("--" * 30)
    print("CNN Text Recognition: Ramen Ratings (Good vs Ordinary)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    texts, labels, classes = load_data(DATA_PATH, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total reviews : {len(texts)}")
    print(f"Classes       : {classes}")

    # Step 2: Split Dataset
    # Text is split BEFORE vectorising: the vocabulary must be built from
    # training data only, otherwise test words leak into the model.
    print("\n[Step 2] Splitting dataset...")

    text_train, text_val, text_test, y_train, y_val, y_test = split_dataset(
        texts, labels, TEST_SIZE, VAL_SIZE
    )

    print(f"Training samples  : {len(text_train)}")
    print(f"Validation samples: {len(text_val)}")
    print(f"Testing samples   : {len(text_test)}")

    # Step 3: Preprocessing (vocabulary + fixed-length sequences)
    print("\n[Step 3] Preprocessing text...")

    vocabulary = build_vocabulary(text_train, MAX_WORDS)
    vocab_size = len(vocabulary) + 2   # + padding and unknown

    X_train = to_features(texts_to_sequences(text_train, vocabulary, MAX_LEN))
    X_val = to_features(texts_to_sequences(text_val, vocabulary, MAX_LEN))
    X_test = to_features(texts_to_sequences(text_test, vocabulary, MAX_LEN))

    save_vocabulary(vocabulary, f"{OUTPUT_DIR}/vocabulary.json")

    np.save(f"{OUTPUT_DIR}/features.npy",
            to_features(texts_to_sequences(texts, vocabulary, MAX_LEN)))
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    # Raw test text is kept so test_cnn.py can print readable reviews
    with open(f"{OUTPUT_DIR}/text_test.json", "w") as f:
        json.dump(list(text_test), f)

    print(f"Feature shape: {X_train.shape}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, vocab_size, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
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