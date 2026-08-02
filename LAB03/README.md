# LAB03 — Regression & Classification

Lab assignment 3: predicting **Age** using Regression and **Gender** using Classification, based on the **Abalone Dataset**.

> **Note on the data:** the original assignment asked for age/gender prediction from **face images** (e.g. UTKFace), but due to the difficulty of accessing the large image files on Kaggle, the **Abalone Dataset** — a real public dataset from the UCI Machine Learning Repository — was used instead. The full machine learning workflow (Data Preparation → PCA → Train/Test → Evaluate → Compare) remains exactly the same; only the features change, from face-image pixels to physical shell measurements.

## 📊 Dataset

[Abalone Dataset (UCI/Kaggle)](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) — physical measurements of 4,177 abalone samples.

| Column | Description |
|---|---|
| `Sex` | Gender: M (Male) / F (Female) / I (Infant) |
| `Length`, `Diameter`, `Height` | Shell dimensions (mm) |
| `Whole_weight`, `Shucked_weight`, `Viscera_weight`, `Shell_weight` | Weight of different parts (g) |
| `Rings` | Number of shell rings → used to compute **Age = Rings + 1.5 (years)** |

Data file: [`data/abalone.csv`](data/abalone.csv)

## 📁 Folder Structure

```
LAB03/
├── data/
│   └── abalone.csv
├── images/                              # plots exported from the notebook
├── notebook
│   └──Lab3_Regression_Classification_TH.html
└── README.md
```

## ⚙️ Setup & How to Run

1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Open `Lab3_Regression_Classification_TH.ipynb` with Jupyter Notebook / JupyterLab / VS Code
3. Run the cells in order (make sure `data/abalone.csv` is in the path referenced by the notebook)

**Libraries used:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (`train_test_split`, `LinearRegression`, `LogisticRegression`, `StandardScaler`, `LabelEncoder`, `PCA`, and various metrics)

`RANDOM_STATE = 42` is set throughout so results are reproducible.

## 🔍 Workflow

### Exploratory Data Analysis (EDA)
- Check for missing values and inspect the distribution of the data
- Visualize the distribution of Age and Sex
- Build a correlation heatmap — `Shell_weight` turns out to have the strongest correlation with `Age`/`Rings` among single features

### LAB 1: Regression (Age Prediction)
1. **Simple Linear Regression** — predict Age from a single feature (`Shell_weight`)
2. **Multiple Linear Regression** — predict Age from all numeric features, including `Sex` after one-hot encoding
3. **PCA + Linear Regression** — reduce the dimensionality of the physical features with PCA before fitting a regression model

### LAB 2: Classification (Gender Prediction)
- Use only the Male/Female subset (Infants excluded) → turns the task into binary classification
- Visualize the decision boundary using the two most relevant features (`Shell_weight`, `Diameter`)
- Fit a **Logistic Regression** model using all features, evaluated with a confusion matrix and ROC curve/AUC

### LAB 3: Model Comparison
- Compare Simple vs. Multiple vs. PCA-based Linear Regression (R², MAE, RMSE, R² gap)
- Compare the conceptual differences between Regression and Classification
- Summarize the metrics of both tasks side by side

## 📈 Results Summary

**Regression (Age Prediction):**
- Multiple Linear Regression clearly outperforms Simple Linear Regression, since it combines information from multiple dimensions
- PCA + Linear Regression reduces the number of features while keeping performance close to Multiple LR

**Classification (Gender Prediction):**
- Logistic Regression can separate genders to some extent, but not very strongly, since the physical characteristics of males and females overlap considerably (unlike Infants, which are more clearly distinguishable)

Full numeric results and plots are available in the notebook ([`.ipynb`](Lab3_Regression_Classification_TH.ipynb) or the exported [`.html`](Lab3_Regression_Classification_TH.html) version).

## 🛠 Tools

- Python 3
- scikit-learn (LinearRegression, LogisticRegression, PCA, StandardScaler, LabelEncoder)
- pandas, numpy for data handling
- matplotlib, seaborn for visualization
