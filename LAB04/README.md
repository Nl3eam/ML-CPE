# LEB 1 — K-Nearest Neighbors (KNN) on the Ramen Ratings Dataset

This project follows Assignment 4 (LEB 1: KNN on a Dataset of Your Choice).
The **Ramen Ratings** dataset from Kaggle was chosen to build a complete
K-Nearest Neighbors (KNN) pipeline: loading the data, cleaning it,
engineering features, scaling, training the model with several k values,
and evaluating with accuracy.

---

## Objectives

1. Understand how KNN works and apply it to a real dataset for classification.
2. Practice building a full pipeline — load data → preprocess → train model
   → evaluate → summarize results — as a complete, working prototype system.

---

## Dataset

**Ramen Ratings Dataset** (Kaggle):
https://www.kaggle.com/datasets/residentmario/ramen-ratings

The data contains instant-noodle (ramen) reviews from The Ramen Rater
website. It has **2,580 rows** before cleaning, with the following columns:

| Column       | Meaning                                                    |
|--------------|-------------------------------------------------------------|
| `Review #`   | Review sequence number (higher = more recent review)       |
| `Brand`      | Ramen brand                                                 |
| `Variety`    | Product name / flavor                                       |
| `Style`      | Packaging style (Pack, Bowl, Cup, Tray, Box, Can)           |
| `Country`    | Country of origin / sale (38 countries in total)            |
| `Stars`      | Review rating (0–5 stars) — some rows contain `"Unrated"`   |
| `Top Ten`    | Year and rank if it made a yearly top-10 list (mostly empty)|

---

## Prediction Task

The raw dataset has no ready-made "class" column to predict, so a target
column was engineered from `Stars`, split into 3 rating levels:

| Class (Level) | Condition          | Meaning        |
|----------------|--------------------|----------------|
| `Low`          | Stars < 2.5        | Not very tasty |
| `Medium`       | 2.5 ≤ Stars < 4.0  | Decent         |
| `High`         | Stars ≥ 4.0        | Very tasty     |

The model predicts which level a given review belongs to, based on
**Style, Country, and how often its Brand appears** in the dataset
(the raw `Stars` value itself is never used as a feature, to avoid
data leakage).

### Features Used

| Feature            | How it is encoded                                                  |
|---------------------|----------------------------------------------------------------------|
| `Style_encoded`     | Label Encoding (Pack, Bowl, Cup, ... → 0, 1, 2, ...)                 |
| `Country_encoded`   | Label Encoding (38 countries → 0–37)                                 |
| `Brand_frequency`   | Frequency Encoding (how many times that brand appears in the data)   |

**Frequency Encoding** is used for Brand instead of One-Hot Encoding
because there are hundreds of distinct brands — one-hot would blow up
the feature dimensionality.

---

## Pipeline

1. **Load** — read `ramen-ratings.csv`
2. **Clean** — drop rows where `Stars == "Unrated"` and rows missing
   Style/Country/Brand (2,575 rows remain)
3. **Feature Engineering** — build the `rating_category` target column
   from `Stars`, and encode the text features as shown above
4. **Split** — stratified 80% train / 20% test split (keeps the class
   proportions the same in both sets)
5. **Scale** — apply `StandardScaler` to put all features on the same
   scale (critical for KNN, since it decides by distance)
6. **Train & Evaluate** — train KNN with k = 3, 5, 7 and measure accuracy
   on the test set
7. **Select best k** — pick the k with the highest test accuracy
8. **Report** — print a classification report, confusion matrix, compare
   against a baseline (always predicting the most common class), and
   save the results to files

---

## Results

From an actual run (`python main.py`):

| k | Test Accuracy |
|---|----------------|
| 3 | 53.6% |
| 5 | 54.6% |
| **7** | **57.5% (best)** |

- **Baseline** (always predict `Medium`) gets **46.6%** accuracy.
- The KNN model with k = 7 beats the baseline by about **11 percentage
  points**, showing that Style / Country / Brand do carry some useful
  signal — but not a huge amount, since taste is ultimately a subjective
  judgement made by the reviewer.

**Confusion Matrix (k = 7):**

|              | Predicted Low | Predicted Medium | Predicted High |
|--------------|:---:|:---:|:---:|
| **True Low**    | 6   | 28  | 16  |
| **True Medium** | 7   | 149 | 84  |
| **True High**   | 3   | 81  | 141 |

The `Low` class is predicted correctly the least often (it also has the
fewest examples in the data — 250 out of 2,575 rows). Most of the
confusion happens between `Medium` and `High`, since their underlying
Stars values are close together (e.g. 3.75 vs 4.0).

---

## Discussion

- **Why is k = 7 the best?** A larger k reduces noise from nearby
  outlier points, making the majority vote more stable. But if k gets
  too large, the model starts to just predict the majority class and
  loses discriminative power.
- **Limitations of the chosen features** — Style, Country, and Brand are
  categorical attributes that don't say much about how a ramen actually
  tastes. Classifying "tastiness" from these alone has a fairly low
  accuracy ceiling.
- **Possible improvements** — extract keywords from the `Variety` column
  (e.g. "Spicy", "Curry", "Tom Yum") as additional features, try other
  values of k beyond 3/5/7, or compare against other algorithms such as
  Decision Tree / Random Forest.

---

## Project Structure

```text
LEB1-KNN-Ramen/
│
├── data-ramen/
│   └── ramen-ratings.csv          ← raw data from Kaggle
│
├── classification/
│   ├── main.py                    ← main script, runs the full pipeline
│   ├── data_loader.py             ← load / clean / encode / split / scale
│   ├── knn_model.py               ← wrapper around KNeighborsClassifier
│   ├── evaluate.py                ← plots + confusion matrix + report
│   └── outputs/
│       ├── 01_k_curve.png         ← accuracy comparison across k values
│       ├── 02_confusion_matrix.png
│       └── predictions.csv        ← actual predictions on the test set
│
├── requirements.txt
├── link-data.txt                  ← link to the original data source
└── README.md
```

---

## How to Run

```bash
# 1) Install required libraries
pip install -r requirements.txt

# 2) Go into the classification folder and run it
cd classification
python main.py
```

The program prints its progress step by step (STEP 1–5) to the terminal
and automatically saves plots/tables into `classification/outputs/`.

---

## Requirements

- `pandas` — table handling / data cleaning
- `numpy` — numerical computation
- `scikit-learn` — `KNeighborsClassifier`, `StandardScaler`,
  `LabelEncoder`, `train_test_split`, and metrics
- `matplotlib` — plotting the k-curve and confusion matrix

---

## Reference

- Dataset: Ramen Ratings (Kaggle) —
  https://www.kaggle.com/datasets/residentmario/ramen-ratings