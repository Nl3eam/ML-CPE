# Microsoft (MSFT) Dataset: 1986 to Present

This project performs a full exploratory data analysis (EDA) and preprocessing pipeline on the Microsoft (MSFT) Stock Dataset from Kaggle (1986 to Present). It covers dataset exploration, visualization, data cleaning, and feature engineering, structured into four labs/parts as required by the assignment.

## Dataset
* **Source:** [Microsoft (MSFT) Dataset: 1986 to Present](https://www.kaggle.com/datasets/jek1wantaufik/microsoft-msft-dataset-1986-to-present?resource=download) (Kaggle)
* **File:** `MSFT_price.csv`
* **Target variable:** `Price_Direction` (1/0 — whether the closing price was higher than the opening price)

## Project Structure
```
MSFT-Data-Preprocessing-Lab2/
├── lab.ipynb
├── lab.py
├── MSFT_price.csv
└── README.md
```
## Requirements
Install the required Python packages before running:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
Open `lab.py` or `lab.ipynb` in VS Code with the Python/Jupyter extensions installed. The code can be executed in two ways:
Interactive/Jupyter Notebook: If running `lab.py`, the file is divided into cells using `# %%` markers. Run each cell sequentially (top to bottom) using the Run Cell button or Interactive Window. For `lab.ipynb`, execute cells from top to bottom since later cells (Part 3 and Part 4) depend on transformations made in earlier cells.
Standard Python Script: Run the entire preprocessing pipeline directly via Terminal using the following command:
```
py -3.11 lab.py
```
## 1. LAB1: Dataset Exploration
* **Load Dataset:** Loaded the raw historical stock data using `pandas.read_csv('MSFT_price.csv')`.
* **Display Shape:** Verified the dataset dimensions showing the total number of rows and columns using `.shape`.
* **Display Data Types:** Inspected the storage data type of each column using ```.dtypes```.
* **Display Summary Statistics:** Checked the statistical metrics (mean, standard deviation, min, max, percentiles) using `.describe()`.
* **Display Missing Values:** Quantified the occurrences of missing data per column using `.isnull().sum()`.
* **Display Duplicate Records:** Checked for redundant rows in the dataset using `.duplicated().sum()`.
* **Display Class Distribution:** Engineered a custom target variable named `Price_Direction` (1 if `Close` > `Open`, else 0) and analyzed its distribution using `.value_counts(normalize=True)`.

## Lab 2: Data Visualization
* **Histogram:** Plotted the distribution of all quantitative numerical features (`Ope`, `High`, `Low`, `Close`, `Volume`) to inspect data skewness.
* **Correlation Heatmap:** Calculated the Pearson correlation matrix and rendered it using `seaborn.heatmap()` to evaluate relationships and check for collinearity among price variables.

## Part 3: Data Cleaning
### Perform:
* **Missing Value Handling:** Provided a strategy to fill missing entries in the `Volume` column using the median value via `.fillna()`.
* **Duplicate Removal:** Identified and dropped potential redundant records using `drop_duplicates()`.
* **Incorrect Data Correction:** Scanned and handled anomalous or misplaced values within the structural fields.
* **Data Type Conversion:** Converted the `Date` column from a string object into a proper `datetime64` format using `pd.to_datetime()`.
### Compare:
* **Mean vs Median:** Computed and compared the mean and median of the `Volume` column. Because the trading volume displays a heavily right-skewed distribution due to high-activity days, the median was chosen over the mean as a more robust central estimate against outliers.

## Part 4: Feature Engineering
* **Label Encoding:** Extracted the month name from the `Date` column and applied `LabelEncoder` from Scikit-Learn to convert categorical month names into ordered sequential integers.
* **One-Hot Encoding:** Extracted the financial quarter from the `Date` column and applied `pd.get_dummies()` with `drop_first=True` to eliminate baseline variables and prevent the Dummy Variable Trap.
* **Feature Scaling:** Applied standard Z-score normalization using `StandardScaler` on pricing features (`Open`, `Close`) to scale the values down to a mean of 0 and variance of 1, which is necessary due to MSFT's exponential growth since 1986.

## Conclusion & Insights
* **Feature Interdependence:** Price features (`Open`, `High`, `Low`, `Close`) display an extremely high linear correlation (close to `1.00`), showing strong collinear relationship, while `Volume`operates with lower baseline linear dependence.
* **Distribution Characteristics:** The trading Volume shows a significant positive skewness (mean is noticeably higher than the median), proving that the median is the most stable metric for handling data anomalies.
* **Model Readiness:** After performing data cleaning, structural One-Hot encoding, and scaling pricing variables, the dataset is 100% complete and optimally prepared for training supervised Time-Series Machine Learning Models.

## Author
Prepared as part of a Machine Learning / Data Preprocessing lab assignment (Lab 1–2, Part 3–4).
