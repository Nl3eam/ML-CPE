from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from data_load import group_rare_categories

STARS_THRESHOLD = 4.0   # "Good" ramen: rated 4 stars or higher
TOP_BRANDS = 30
TOP_COUNTRIES = 15

CLASSES = ["Not Good", "Good"]


def add_label(df):
    """Turn the continuous 'Stars' rating into a binary Good/Not Good label."""

    df = df.copy()
    df["label"] = (df["Stars"] >= STARS_THRESHOLD).astype(int)
    return df


def engineer_features(df):
    """Reduce high-cardinality text columns to something one-hot friendly.

    Returns a DataFrame with the columns the model pipeline expects:
    Brand (grouped), Style, Country (grouped), Variety (free text).
    """

    df = df.copy()
    df["Brand"] = group_rare_categories(df["Brand"], TOP_BRANDS)
    df["Country"] = group_rare_categories(df["Country"], TOP_COUNTRIES)

    return df[["Brand", "Style", "Country", "Variety"]]


def build_feature_pipeline():
    """Unfitted ColumnTransformer: one-hot categoricals + TF-IDF on Variety.

    Mirrors the role preprocess.to_features() played for pixel data in the
    original image pipeline, but must be *fit on the training split only*
    (handled inside svm_model.train_svm) to avoid leaking test data.
    """

    return ColumnTransformer(
        transformers=[
            ("brand", OneHotEncoder(handle_unknown="ignore"), ["Brand"]),
            ("style", OneHotEncoder(handle_unknown="ignore"), ["Style"]),
            ("country", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
            # A single string column name (not a list) makes ColumnTransformer
            # pass a 1-D text series into TfidfVectorizer, as it expects.
            ("variety", TfidfVectorizer(max_features=150, stop_words="english"), "Variety"),
        ],
        sparse_threshold=0.0,  # force a dense matrix (PCA/SVC need dense input)
    )
