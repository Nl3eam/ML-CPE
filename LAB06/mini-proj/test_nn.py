"""Test the trained model on random rows from the test set.

Random sample every run. Run main.py first.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 8


def test_nn(n_samples=N_SAMPLES):

    model = keras.models.load_model(f"{OUTPUT_DIR}/nn_model.keras")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    meta = pd.read_csv(f"{OUTPUT_DIR}/test_meta.csv")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]
    meta_sample = meta.iloc[index].reset_index(drop=True)

    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    print(f"{'Brand':<15}{'Style':<8}{'Country':<12}{'Stars':<7}"
          f"{'Pred':<20}{'True':<20}{'Conf':<7}Result")
    correct_total = 0
    for i in range(n_samples):
        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        correct_total += int(correct)
        row = meta_sample.iloc[i]

        print(f"{row['Brand']:<15.15}{row['Style']:<8.8}{row['Country']:<12.12}"
              f"{row['Stars']:<7}{pred:<20}{true:<20}"
              f"{confidence[i] * 100:5.1f}%  {'OK' if correct else 'WRONG'}")

    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Correct", "Wrong"],
           [correct_total, n_samples - correct_total],
           color=["#4caf50", "#e53935"])
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction sample: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_nn()
