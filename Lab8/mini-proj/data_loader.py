import os

import pandas as pd

# Fixed order so "Low < Medium < High" is kept in the confusion matrix
RATING_ORDER = ["Low", "Medium", "High"]


def rating_bucket(stars):
    if stars < 3.0:
        return "Low"
    if stars < 4.0:
        return "Medium"
    return "High"


def load_data(csv_path, target="Style", min_per_class=30):
    """Read ramen-ratings.csv and return (texts, labels, classes).

    target = "Style"  -> predict Pack / Bowl / Cup / Tray
    target = "Rating" -> predict Low / Medium / High from Stars
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(csv_path)}\n"
            "Download ramen-ratings.csv from "
            "https://www.kaggle.com/datasets/residentmario/ramen-ratings "
            "and put it in the data/ folder."
        )

    df = pd.read_csv(csv_path)
    print(f"Rows in CSV: {len(df)}")

    # The text the network reads: Brand | Variety | Country
    df["text"] = (df["Brand"].astype(str) + " | "
                  + df["Variety"].astype(str).str.strip() + " | "
                  + df["Country"].astype(str))

    if target == "Rating":
        # Stars contains the string "Unrated" -> becomes NaN and is dropped
        df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce")
        df = df.dropna(subset=["Stars"])
        df["label_name"] = df["Stars"].apply(rating_bucket)
    elif target == "Style":
        df = df.dropna(subset=["Style"])
        df["label_name"] = df["Style"]
    else:
        raise ValueError("target must be 'Style' or 'Rating'")

    # Drop very rare classes (e.g. Box, Can, Bar) - too few to learn or split
    counts = df["label_name"].value_counts()
    keep = counts[counts >= min_per_class].index
    dropped = counts[counts < min_per_class]
    if len(dropped):
        print("Dropped rare classes:", dict(dropped))
    df = df[df["label_name"].isin(keep)]

    if target == "Rating":
        classes = [c for c in RATING_ORDER if c in keep]
    else:
        classes = sorted(keep)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = df["label_name"].map(class_to_idx).to_numpy()
    texts = df["text"].tolist()

    for c in classes:
        print(f"Loaded class {c}: {(labels == class_to_idx[c]).sum()} rows")

    return texts, labels, classes