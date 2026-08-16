import json
import os

import joblib

from data_load import load_data
from preprocess import add_label, engineer_features, CLASSES, STARS_THRESHOLD
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model
from visualize import (
    plot_stars_distribution,
    plot_style_and_country,
    plot_pca_scatter,
    plot_roc_curve,
)

DATA_PATH = "data/ramen-ratings.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2


def main():

    print("--" * 30)
    print("SVM Ramen Ratings: Good vs Not Good")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Total reviews : {len(df)}")

    # Step 2: Build features + label
    print("\n[Step 2] Building features and labels...")
    df = add_label(df)
    X = engineer_features(df)         # DataFrame: Brand, Style, Country, Variety
    y = df["label"].to_numpy()

    print(f"Classes       : {CLASSES}  (Stars >= {STARS_THRESHOLD} = 'Good')")
    print(f"Class balance : {y.mean() * 100:.1f}% Good")

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(CLASSES, f)

    plot_stars_distribution(df, f"{OUTPUT_DIR}/stars_distribution.png")
    plot_style_and_country(df, f"{OUTPUT_DIR}/style_and_country.png")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    X_train.to_pickle(f"{OUTPUT_DIR}/X_train.pkl")
    X_test.to_pickle(f"{OUTPUT_DIR}/X_test.pkl")
    joblib.dump(y_train, f"{OUTPUT_DIR}/y_train.pkl")
    joblib.dump(y_test, f"{OUTPUT_DIR}/y_test.pkl")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4: Train SVM
    print("\n[Step 4] Training SVM...")

    model, scaler = train_svm(X_train, y_train)

    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print("SVM training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_svm(model, scaler, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, CLASSES,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")

    plot_roc_curve(model, scaler, X_test, y_test, f"{OUTPUT_DIR}/roc_curve.png")
    plot_pca_scatter(scaler, X_test, y_test, f"{OUTPUT_DIR}/pca_scatter.png")


if __name__ == "__main__":
    main()
