import os

import numpy as np
import pandas as pd

from preprocessing import preprocess_text

# Columns we expect in the Kaggle file
TEXT_COLUMNS = ["Brand", "Variety", "Style", "Country"]
STAR_COLUMN = "Stars"

# A bowl with >= 4 stars is treated as "Good"
GOOD_THRESHOLD = 4.0
CLASS_NAMES = ["Ordinary", "Good"]


def load_data(data_path, max_per_class=None):
    """Read ramen-ratings.csv and return (texts, labels, classes).

    Same role as load_data() in the image version: it walks the raw
    dataset, throws away anything unusable, and hands back one flat
    array of samples plus their integer labels.
    """

    # Fail with a useful message instead of a bare pandas error
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "Expected data/ramen-ratings.csv at the project root. Download it "
            "from kaggle.com/datasets/residentmario/ramen-ratings, or point "
            "DATA_PATH in main.py at an existing copy."
        )

    frame = pd.read_csv(data_path)

    missing = [c for c in TEXT_COLUMNS + [STAR_COLUMN]
               if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # "Unrated" and blanks become NaN, then get dropped
    stars = pd.to_numeric(frame[STAR_COLUMN], errors="coerce")

    texts = []
    labels = []
    skipped = 0

    for row, star in zip(frame[TEXT_COLUMNS].itertuples(index=False), stars):
        if pd.isna(star):
            skipped += 1
            continue

        # Join the columns into one review string:
        # "nissin cup noodles seafood pack japan"
        raw = " ".join("" if pd.isna(v) else str(v) for v in row)
        text = preprocess_text(raw)

        # Skip empty or unusable rows
        if text is None:
            skipped += 1
            continue

        texts.append(text)
        labels.append(1 if star >= GOOD_THRESHOLD else 0)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=np.int64)

    print("Detected classes:", CLASS_NAMES)

    # Optionally cap each class so the two classes stay balanced
    if max_per_class:
        rng = np.random.default_rng(42)
        keep = []
        for label in range(len(CLASS_NAMES)):
            index = np.flatnonzero(labels == label)
            if len(index) > max_per_class:
                index = rng.choice(index, max_per_class, replace=False)
            keep.append(index)
        keep = np.sort(np.concatenate(keep))
        texts, labels = texts[keep], labels[keep]

    for label, class_name in enumerate(CLASS_NAMES):
        print(f"Loaded class {class_name}: {int((labels == label).sum())} rows")
    print(f"Skipped rows (unrated / empty): {skipped}")

    return texts, labels, CLASS_NAMES