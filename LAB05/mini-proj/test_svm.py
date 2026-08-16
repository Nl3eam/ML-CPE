import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"
N_SAMPLES = 8


def test_svm(n_samples=N_SAMPLES):

    # Load model and test set
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = pd.read_pickle(f"{OUTPUT_DIR}/X_test.pkl")
    y_test = joblib.load(f"{OUTPUT_DIR}/y_test.pkl")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # Pick random reviews (no seed -> different every run)
    n_samples = min(n_samples, len(X_test))
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test.iloc[index]
    y_sample = y_test[index]

    # Predict
    predictions = model.predict(scaler.transform(X_sample))

    fig, ax = plt.subplots(figsize=(9, 0.6 * n_samples + 1))
    ax.axis("off")

    correct_total = 0
    for row, (_, sample) in enumerate(X_sample.iterrows()):
        pred = classes[predictions[row]]
        true = classes[y_sample[row]]
        correct = predictions[row] == y_sample[row]
        correct_total += int(correct)
        color = "green" if correct else "red"

        label = f"{sample['Brand']} — {sample['Variety']} ({sample['Style']}, {sample['Country']})"
        line = f"Pred: {pred:<9} True: {true:<9} {'OK' if correct else 'WRONG'}"

        y_pos = 1 - (row + 0.5) / n_samples
        ax.text(0.0, y_pos, label[:70], fontsize=9, va="center")
        ax.text(0.62, y_pos, line, fontsize=9, va="center", color=color, family="monospace")

        print(f"[{row + 1}] {label}\n    {line}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"\nCorrect: {correct_total}/{n_samples}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_svm()
