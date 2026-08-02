# Wine Quality Prediction — Regression & Classification

Predicting **wine quality** (regression) and **good vs. not-good wine** (classification) from physicochemical measurements, using the Red Wine Quality dataset (Cortez et al., 2009).

This project was built as Worksheet 3 (Regression & Classification), covering Data Preprocessing, Regression, Classification, and Model Comparison in a single notebook.

---

## Objectives

1. **Understand Regression and Classification Principles:** Comprehend the foundational concepts of Supervised Learning and explain the differences between predicting continuous values and categorical classification.
2. **Data Preparation and Dimensionality Reduction:** Prepare datasets for modeling by selecting appropriate features and applying Principal Component Analysis (PCA) to reduce dimensionality and improve efficiency.
3. **Develop Regression and Classification Models:** Build a Linear Regression model to predict wine quality score, and a Classification model to predict good vs. not-good wine, comparing the trade-offs of each approach.
4. **Python Programming and Machine Learning Workflow:** Write Python code using machine learning libraries to train, test, and evaluate models.
5. **Evaluate and Interpret Performance Metrics:** Analyze and interpret results using appropriate metrics — Accuracy, Precision, Recall, F1-score, ROC Curve, and AUC — and present the work on GitHub as part of a portfolio.

---

## Dataset

- **Source:** [Red Wine Quality (Cortez et al., 2009) — Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (original: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality))
- **Samples:** 1,599 red wine samples
- **Features (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Regression target:** `quality` — a sensory score from 0-10 given by wine tasters
- **Classification target:** `good_wine` — derived label, 1 if `quality >= 7` ("good"), else 0

---

## Laboratory Structure & Notebook Contents

### LAB 1: Regression
* **Simple Linear Regression:** Modeling the linear relationship between a single feature and wine quality.
* **Multiple Linear Regression:** Developing a multivariable regression model using multiple features combined.
* **Quality Prediction:** Predicting continuous wine quality scores and evaluating models using $R^2$, MAE, and RMSE.

### LAB 2: Classification
* **Preparing Classification Data:** Transforming target labels into binary classes (`good_wine` vs. others).
* **Decision Boundary Visualization:** Visualizing the decision boundaries of classification models.
* **Logistic Regression:** Implementing logistic regression for categorical prediction.
* **Wine Category Prediction:** Predicting and classifying wine quality groups.
* **Confusion Matrix:** Analyzing performance errors across classes.

### LAB 3: Model Comparison
* **Simple vs Multiple Linear Regression:** Comparing performance between single-feature and multi-feature regression models.
* **Training vs Testing Performance:** Analyzing generalization, overfitting, and underfitting.
* **Regression vs Classification:** Contrasting continuous prediction versus discrete classification approaches.
* **Model Performance Metrics:** Summarizing metrics including Accuracy, Precision, Recall, F1-score, ROC Curve, and AUC.

---

## Project Structure

```text
LAB03/
├── data/
│   └── winequality-red.csv
├── image/
│   ├── 01_eda_distributions.png
│   ├── 02_pca_variance.png
│   ├── 03_simple_linear_regression.png
│   ├── 04_actual_vs_predicted_quality.png
│   ├── 05_decision_boundary.png
│   ├── 06_confusion_matrix.png
│   └── 07_roc_curve.png
├── notebook/
│   └── wine_quality_regression_classification_lab.ipynb
└── README.md
```

---

## Which CSV should you submit?

Use the **real dataset you downloaded from Kaggle/UCI** (`winequality-red.csv`) — that's the one that belongs in `data/` and the one you submit. The other file, `_demo_winequality.csv`, is auto-generated **synthetic placeholder data** the notebook creates on its own only when it can't find the real CSV, purely so every cell can still run and be checked for errors. Its numbers are random and carry no real meaning about wine quality, so don't submit it — delete it (or just make sure the real CSV is in `data/` before your final run, so the notebook never falls back to demo mode).

---

## How to Run

1. Download `winequality-red.csv` from the Kaggle link above (or the UCI repository) and place it in the `data/` folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Open `notebook/wine_quality_regression_classification_lab.ipynb` in Jupyter, VS Code, Kaggle, or Google Colab, and run all cells (Restart & Run All).
   - If the CSV isn't found in `data/`, the notebook automatically falls back to a synthetic **DEMO MODE** dataset so every cell can still run — swap in the real CSV for meaningful results (see note above).
4. All plots are automatically saved to `../figures/` as you run the notebook.

---

## Key Results

> Replace this table with the numbers from your own run once you have the real dataset.

| Task | Metric | Score |
|---|---|---|
| Regression (Quality) | MAE / RMSE / R² | — |
| Classification (Good/Bad Wine) | Accuracy / Precision / Recall / F1 | — |
| Classification (Good/Bad Wine) | AUC | — |

**Notes on the dataset:** "Good" wines (quality ≥ 7) are a small minority of the data, so Accuracy alone can be misleading — Precision, Recall, F1, and AUC give a fairer picture of classification performance. Wine quality is a subjective sensory score shaped by many interacting chemical factors, so don't expect a very high R² even from the best regression model — a modest R² with a clear discussion of *why* is a perfectly valid, honest result.

---

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn (LinearRegression, LogisticRegression, PCA, StandardScaler, metrics)
- Matplotlib

---

## Author

Lab 3: Regression & Classification — Machine Learning course
