import numpy as np
import pandas as pd

RATING_THRESHOLD = 4.0   
TOP_BRANDS = 30           
TOP_COUNTRIES = 20        


def make_labels(df):
    """1 if a variety is highly rated (Stars >= threshold), else 0."""

    labels = (df["Stars"] >= RATING_THRESHOLD).astype(int).to_numpy()
    classes = ["Not Highly Rated", "Highly Rated"]
    return labels, classes


def _group_rare(series, top_n):
    """Keep the top_n most frequent categories, fold the rest into 'Other'.

    Brand has ~350 unique values and Country has ~40; one-hot encoding all
    of them would make the feature vector huge and mostly empty per row.
    """

    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), "Other")


def to_features(df, fit_columns=None):
    """One-hot encode Brand / Style / Country into a numeric feature matrix.

    If fit_columns is given (from the training set), the output is aligned
    to that exact column set so train/val/test all share the same shape,
    the tabular equivalent of resizing every image to the same resolution.
    """

    features = pd.DataFrame({
        "Brand": _group_rare(df["Brand"].fillna("Unknown"), TOP_BRANDS),
        "Style": df["Style"].fillna("Unknown"),
        "Country": _group_rare(df["Country"].fillna("Unknown"), TOP_COUNTRIES),
    })

    encoded = pd.get_dummies(features, columns=["Brand", "Style", "Country"])

    if fit_columns is not None:
        encoded = encoded.reindex(columns=fit_columns, fill_value=0)

    encoded = encoded.astype(np.float32)

    return encoded.to_numpy(), list(encoded.columns)
