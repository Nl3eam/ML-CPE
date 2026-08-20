import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import make_labels, to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ramen-ratings.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.2
VAL_SIZE = 0.1
EPOCHS = 30
BATCH_SIZE = 32


def main():

    print("--" * 30)
    print("Neural Network Tabular Classification: Ramen Rating (High vs Not)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    df = load_data(DATA_PATH)
    labels, classes = make_labels(df)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total rows : {len(df)}")
    print(f"Classes    : {classes}")


    print("\n[Step 2] Splitting dataset...")

    row_index = np.arange(len(df)).reshape(-1, 1)
    idx_train, idx_val, idx_test, y_train, y_val, y_test = split_dataset(
        row_index, labels, TEST_SIZE, VAL_SIZE
    )

    df_train = df.iloc[idx_train.ravel()]
    df_val = df.iloc[idx_val.ravel()]
    df_test = df.iloc[idx_test.ravel()]

    print(f"Training rows  : {len(df_train)}")
    print(f"Validation rows: {len(df_val)}")
    print(f"Testing rows   : {len(df_test)}")

    print("\n[Step 3] Encoding features...")

    X_train, feature_columns = to_features(df_train)
    X_val, _ = to_features(df_val, fit_columns=feature_columns)
    X_test, _ = to_features(df_test, fit_columns=feature_columns)

    with open(f"{OUTPUT_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)


    df_test[["Brand", "Variety", "Style", "Country", "Stars"]].to_csv(
        f"{OUTPUT_DIR}/test_meta.csv", index=False
    )

    print(f"Feature shape: {X_train.shape}")

    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()
