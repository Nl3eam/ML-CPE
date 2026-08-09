"""
LEB 1 : KNN on a Dataset of Your Choice
Dataset  : Ramen Ratings (Kaggle) - https://www.kaggle.com/datasets/residentmario/ramen-ratings
Task     : classify a ramen review into a rating category (Low / Medium / High)
           based on Style, Country and Brand popularity.
"""

from pathlib import Path

import numpy as np

import data_loader
import evaluate
from knn_model import KNNModel

OUT_DIR = Path(__file__).resolve().parent / "outputs"

K_VALUES = [3, 5, 7]

def title(text):
    print("\n" + "--" * 30)
    print(text)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load and preprocess data")

    data = data_loader.load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    print(f"data of all (after cleaning) : {data['n_rows']} rows")
    print(f"features used : {data['feature_names']}")
    print(f"classes of predictions : {class_names}")
    print(f"split data : train {len(y_train)} / test {len(y_test)}")

    title("STEP 2 : train KNN with k = 3, 5, 7 and evaluate accuracy")

    scores = []
    for k in K_VALUES:
        model = KNNModel(k=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        scores.append(acc)
        print(f"   k = {k}  ->  test accuracy = {acc:.4f}  ({acc * 100:.1f}%)")

    best_k = K_VALUES[int(np.argmax(scores))]
    best_acc = max(scores)
    print(f"\n>>> best k value based on the test accuracy : k = {best_k} "
          f"(accuracy = {best_acc:.4f})")

    evaluate.plot_k_curve(K_VALUES, scores, OUT_DIR / "01_k_curve.png")

    title(f"STEP 3 : re-train with best k = {best_k} and look closer at the result")

    best_model = KNNModel(k=best_k)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("classification report:\n")
    evaluate.print_report(y_test, y_pred, class_names)

    cm = evaluate.plot_confusion_matrix(y_test, y_pred, class_names,
                                        OUT_DIR / "02_confusion_matrix.png")
    print("Confusion Matrix (rows = true label, columns = predicted label):")
    print(cm)

    title("STEP 4 : is our model better than guessing?")

    majority = np.bincount(y_train).argmax()
    baseline = float(np.mean(y_test == majority))

    print(f"Baseline (predict '{class_names[majority]}' every time) : {baseline:.4f}")
    print(f"KNN (k = {best_k})                                     : {best_acc:.4f}")

    title("STEP 5 : save predictions to CSV")

    evaluate.save_predictions(y_test, y_pred, class_names,
                              OUT_DIR / "predictions.csv")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")

    title("Discussion of the experimental results")
    print(f"""
    - accuracy scores for each k : {dict(zip(K_VALUES, [round(s, 4) for s in scores]))}
    - best k value based on the test accuracy : k = {best_k} ({best_acc*100:.1f}%)
    - Style, Country and Brand-frequency give some signal, but ramen ratings
      are ultimately a taste judgement, so accuracy is modest, only a bit
      above the majority-class baseline ({baseline*100:.1f}%).
    - the "Medium" class overlaps a lot with "High" (both are common star
      values like 3.5-4.0), so most mistakes happen between those two.
    - a larger k tends to smooth the decision boundary and reduce noise,
      but if k is too large the model just predicts the majority class.
    """)


if __name__ == "__main__":
    main()
