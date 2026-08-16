# ML-05-SVM (revised) — Ramen Ratings

Revised from [aproot-en/Machine-Learning-Course · ML-05-SVM](https://github.com/aproot-en/Machine-Learning-Course/tree/main/ML-05-SVM),
which built an SVM image classifier for cats vs dogs. This version keeps the
same pipeline shape (load → feature engineer → split → train SVM → evaluate →
predict) but swaps in a **tabular text/categorical dataset**: Kaggle's
[Ramen Ratings](https://www.kaggle.com/datasets/residentmario/ramen-ratings).

## Data

Ramen Ratings dataset (~2,580–3,180 reviews depending on export), columns:
`Review #, Brand, Variety, Style, Country, Stars, Top Ten`.

Download `ramen-ratings.csv` from Kaggle (requires a free account) and place
it at `mini-proj/data/ramen-ratings.csv`:
https://www.kaggle.com/datasets/residentmario/ramen-ratings

## Task

The original Stars rating (0–5) is turned into a **binary classification
problem**, mirroring the Cat/Dog structure of the original project:

- **Good**: Stars ≥ 4.0
- **Not Good**: Stars < 4.0

Features used:
- **Brand** (top 30 by frequency, rest grouped as "Other") — one-hot
- **Style** (Pack / Bowl / Cup / Tray / Box / ...) — one-hot
- **Country** (top 15, rest grouped as "Other") — one-hot
- **Variety** (the product name text, e.g. "Spicy Tom Yum Shrimp") — TF-IDF,
  since flavor keywords carry real signal about rating

## Structure

```text
ramen-svm/
│
├── mini-proj/
│   ├── data/
│   │   └── ramen-ratings.csv      <- place the Kaggle CSV here
│   │
│   └── outputs/
│   │   ├── classes.json
│   │   ├── confusion_matrix.png
│   │   ├── pca_scatter.png         <- 2D PCA view of Good vs Not Good separability
│   │   ├── prediction_sample.png
│   │   ├── roc_curve.png           <- ROC curve on the held-out test set
│   │   ├── stars_distribution.png  <- Stars histogram + Good/Not Good cutoff
│   │   └── style_and_country.png   <- review counts by Style, avg Stars by Country
│   │
│   ├── data_load.py
│   ├── evaluate.py
│   ├── main.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── test_svm.py
│   ├── svm_model.py
│   └── visualize.py               <- extra EDA / model-analysis plots
│
└── README.md
```

## How to run

```bash
cd mini-proj
pip install pandas scikit-learn joblib matplotlib
python main.py        # trains and evaluates the SVM
python test_svm.py    # shows predictions on a few random held-out reviews
```

## Summary

The project uses SVM for binary "Good vs Not Good" ramen rating prediction.
Brand, Style and Country are one-hot encoded; the Variety product name is
turned into TF-IDF text features. Everything is scaled and PCA-reduced
inside a single fitted pipeline, then fed to an RBF-kernel SVM. The model is
evaluated with accuracy, precision, recall, F1-score, and a confusion matrix
— the same evaluation code as the original image-classification project,
since it was already written generically over class labels.
