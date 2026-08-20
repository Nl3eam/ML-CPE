# ML-06-Neural Network (NN) — Ramen Ratings Edition

This project adapts the `ML-06-NN` lesson (originally a Cat vs Dog image classifier) to work with **tabular data** instead — specifically the [Ramen Ratings](https://www.kaggle.com/datasets/residentmario/ramen-ratings) dataset on Kaggle. The overall pipeline (load data → preprocess → split → train NN → evaluate → test) stays identical to the original. What changes is *how the input data is prepared* and *the first layer of the model*, so it fits tabular data instead of images.

---

## 1. About the Dataset

**Ramen Ratings** is a dataset of instant ramen reviews from around the world, originally collected from The Ramen Rater website and published on Kaggle by user `residentmario`.

| Column | Type | Description |
|---|---|---|
| `Review #` | number | Review sequence number; higher = more recent |
| `Brand` | categorical | Ramen brand, e.g. Nissin, Nongshim, Mama (350+ unique brands) |
| `Variety` | free text | The product's name/flavor |
| `Style` | categorical | Packaging format — Pack, Cup, Bowl, Tray, Box, Bar |
| `Country` | categorical | Country of manufacture/sale (~38 countries) |
| `Stars` | number (0.0–5.0) | Review score; some rows have the value `"Unrated"` |
| `Top Ten` | text | Year's Top 10 ranking (mostly blank — under 2% of rows have a value) |

**Raw dataset size:** 2,580 rows
**After cleaning** (dropping rows where `Stars = Unrated` or `Style`/`Country`/`Brand` is missing): 2,575 rows (only 5 rows dropped)

> `mini-proj/data/ramen-ratings.csv` in this project is the actual downloaded dataset, ready to use as-is.

---

## 2. Problem Definition

The original lesson was a **binary classification** task (cat vs dog), which let the model architecture and evaluation code be reused almost as-is. This project keeps it binary too:

> **Predict whether a given ramen variety will receive a "high" review score (Stars ≥ 4.0) or "not high" (Stars < 4.0), using only its brand (Brand), packaging style (Style), and country of origin (Country)** — not the `Variety` text or the `Stars` value itself (since that's the prediction target).

Why a 4.0 threshold:
- The dataset's average score is around 3.6/5, so 4.0+ genuinely represents the upper tier.
- It splits the data into two groups that aren't too imbalanced (see class breakdown below), so a plain metric like accuracy still means something without needing class-weighting tricks.

**Class balance (from the real, cleaned 2,575-row dataset):**
| Class | Count | Share |
|---|---|---|
| Not Highly Rated (Stars < 4.0) | 1,448 | 56.2% |
| Highly Rated (Stars ≥ 4.0) | 1,127 | 43.8% |

---

## 3. Project Structure

```text
ML-06-NN/
│
├── README.md                      
├── requirements.txt                # required libraries
│
└── mini-proj/
    │
    ├── data/
    │   └── ramen-ratings.csv       # raw dataset from Kaggle (2,580 rows)
    │
    ├── data_loader.py              # load the CSV + clean the Stars column
    ├── preprocessing.py            # build the label + one-hot encode Brand/Style/Country
    ├── split_data.py               # split into train / validation / test (stratified)
    ├── nn_model.py                 # build, train, save, and predict with the neural network
    ├── evaluate.py                 # accuracy, classification report, confusion matrix, plots
    ├── test_nn.py                  # test the trained model on random rows from the test set
    ├── main.py                     # main script — runs the full pipeline end to end
    │
    └── outputs/                    # results from an actual run (usable as-is)
        ├── nn_model.keras          # the trained model
        ├── history.json            # loss/accuracy history during training
        ├── classes.json            # class names ["Not Highly Rated", "Highly Rated"]
        ├── feature_columns.json    # one-hot vector column names (for aligning features at inference time)
        ├── test_meta.csv           # Brand/Variety/Style/Country/Stars for test-set rows (so test_nn.py can print readable output)
        ├── confusion_matrix.png    # confusion matrix plot
        ├── training_history.png    # accuracy/loss curves during training
        └── prediction_sample.png   # summary chart of test_nn.py's sample predictions
```

---

## 4. File-by-File Walkthrough

### `data_loader.py`
- Reads the CSV with pandas
- Converts the `Stars` column to numeric; values that can't be converted (e.g. `"Unrated"`) become `NaN` and get dropped
- Drops rows missing `Style`, `Country`, or `Brand`
- Prints a summary of how many rows loaded successfully vs. were skipped (the tabular equivalent of skipping corrupted image files in the original)

### `preprocessing.py`
- **`make_labels(df)`** — builds the binary label from the condition `Stars >= 4.0`
- **`to_features(df, fit_columns=None)`** — one-hot encodes the categorical columns:
  - `Brand`: keeps the 30 most common brands, everything else is grouped into `"Other"` (there are 350+ unique brands — keeping all of them would make the feature vector huge and mostly sparse)
  - `Country`: keeps the 20 most common countries, everything else grouped into `"Other"`
  - `Style`: keeps every value (only about 6 distinct values)
  - When `fit_columns` is passed in (the columns from the training set), the function aligns the validation/test set's columns to match exactly — this prevents data leakage and column-count mismatches between splits

### `split_data.py`
- Performs a stratified split into train / validation / test (default 70% / 10% / 20%)
- Uses the exact same code as the original, since this function is data-type agnostic

### `nn_model.py`
- Multi-Layer Perceptron (MLP) architecture:
  ```
  Input (57 dimensions, one-hot vector)
    → Dense(256, relu) → BatchNorm → Dropout(0.4)
    → Dense(128, relu) → BatchNorm → Dropout(0.4)
    → Dense(64,  relu) → Dropout(0.3)
    → Dense(1, sigmoid)          # binary output
  ```
- Differs from the original by **dropping the `Rescaling(1/255)` and `Flatten` layers**, since tabular data is already a 1D one-hot vector — no pixel scaling or unrolling from a 2D image is needed
- Uses `EarlyStopping` (stops when val_loss stops improving for 5 epochs) and `ReduceLROnPlateau`, exactly as in the original

### `evaluate.py`
- Computes accuracy, a classification report (precision/recall/F1), a confusion matrix, and plots — uses the exact same code as the original, since these are generic functions that don't depend on the data type

### `main.py`
Execution order:
1. Load and clean the data
2. **Split row indices into train/val/test first**, then one-hot encode only the training rows (this avoids the data leakage that would happen if "top brands" were fit on the full dataset, test rows included)
3. Encode the val/test features to match the training set's columns
4. Train the model
5. Predict and evaluate on the test set
6. Save all plots and result files to `outputs/`

### `test_nn.py`
- Samples random rows from the test set (8 by default), predicts on them, and prints a table (Brand, Style, Country, true Stars, predicted class, true class, confidence, correct/wrong)
- Differs from the original, which displayed results as a 2×2 image grid — since tabular data has no images to show, this prints a text table instead, plus a bar chart summarizing correct vs. wrong counts

---

## 5. How to Run

```bash
# Install required libraries
pip install -r requirements.txt

# Run the full pipeline (load data -> train -> evaluate)
cd mini-proj
python main.py

# Test the trained model on random rows from the test set
python test_nn.py
```

> The `data/` and `outputs/` folders in this project already contain files from an actual run. To re-run everything from scratch, delete the files in `outputs/` and run `python main.py` again.

---

## 6. Actual Results (from this project's run)

| Item | Value |
|---|---|
| Rows after cleaning | 2,575 |
| Train / Validation / Test | 1,802 / 258 / 515 rows |
| One-hot feature vector size | 57 dimensions |
| Epochs actually trained (before EarlyStopping kicked in) | 20 of 30 |
| Test Accuracy | **67.18%** |

**Classification Report (Test set):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not Highly Rated | 0.74 | 0.64 | 0.69 | 290 |
| Highly Rated | 0.61 | 0.71 | 0.65 | 225 |

**Confusion Matrix:**
```
                    Predicted
                Not High   Highly Rated
True Not High     186          104
True Highly        65          160
```

The model catches "Highly Rated" items with a decent recall (71%), but precision is more modest (61%) — meaning that Brand/Style/Country alone aren't quite enough to pin down the score precisely. That makes sense, since the actual review score depends a lot on the specific recipe's flavor (captured in `Variety`), which this model doesn't currently use as a feature.

---

## 7. Limitations and Possible Improvements

- **`Variety` text isn't used**, even though it may contain meaningful keywords (e.g. "Spicy", "Tom Yum", "Premium") that correlate with score — this could be added as a text feature (bag-of-words / embeddings) in the future
- **Infrequent brands/countries are grouped into "Other"**, losing detail for brands/countries with few reviews
- **The 4.0 threshold is a fixed choice** — worth experimenting with other thresholds, or reframing the problem as regression (predicting the `Stars` value directly) instead of binary classification
- **No cross-validation** was used, to keep the code as simple and close to the original lesson's style as possible

---

## 8. Comparison with the Original (Cat vs Dog)

| Pipeline stage | Original (images) | This version (tabular) |
|---|---|---|
| Data source | `PetImages/` image folder | `ramen-ratings.csv` file |
| Data loading | Read and resize images with OpenCV | Read CSV with pandas |
| Preprocessing | Resize images + convert BGR→RGB | One-hot encode Brand/Style/Country |
| Model input | Image (H, W, 3) | 1D one-hot vector |
| Model's first layer | `Rescaling` + `Flatten` | None (data is already ready to use) |
| Remaining model layers | Identical | Identical |
| Evaluation | Identical | Identical |
| Data splitting | Identical | Identical |
| Test-time display | 2×2 image grid | Text table + bar chart |
