"""
Read CSV
clean missing / invalid rows
build the target (rating_category) from Stars
convert text (Brand, Style, Country) to numbers
make Scaling for KNN
split data: train / test
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CSV_PATH = Path(__file__).resolve().parent.parent / "data-ramen" / "ramen-ratings.csv"


def stars_to_category(stars):
    if stars < 2.5:
        return "Low"
    elif stars < 4.0:
        return "Medium"
    else:
        return "High"

def load_data(test_size=0.2, seed=42):

    df = pd.read_csv(CSV_PATH)

    df = df[df["Stars"] != "Unrated"].copy()
    df["Stars"] = df["Stars"].astype(float)
    df = df.dropna(subset=["Style", "Country", "Brand"])

    df["rating_category"] = df["Stars"].apply(stars_to_category)
    class_names = ["Low", "Medium", "High"]
    y = df["rating_category"].map({name: i for i, name in enumerate(class_names)})
    y = y.to_numpy(dtype="int32")

    style_enc = LabelEncoder()
    country_enc = LabelEncoder()

    style_num = style_enc.fit_transform(df["Style"])
    country_num = country_enc.fit_transform(df["Country"])

    brand_freq = df["Brand"].map(df["Brand"].value_counts())

    X = pd.DataFrame({
        "Style_encoded": style_num,
        "Country_encoded": country_num,
        "Brand_frequency": brand_freq.to_numpy(),
    })

    feature_names = list(X.columns)
    X = X.to_numpy(dtype="float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": feature_names,
        "n_rows": len(df),
        "style_encoder": style_enc,
        "country_encoder": country_enc,
    }


if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("test  :", data["X_test"].shape)
    print("classes  :", data["class_names"])
