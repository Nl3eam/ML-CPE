import pandas as pd

REQUIRED_COLUMNS = ["Brand", "Variety", "Style", "Country", "Stars"]


def load_data(csv_path):
    """Load the raw Ramen Ratings CSV and drop rows that can't be used.

    A few rows have Stars == "Unrated" and a handful are missing Style,
    so those are dropped here rather than silently coerced (mirrors
    skipping unreadable images in the image-classification version).
    """

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    total = len(df)

    df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce")
    df = df.dropna(subset=["Stars", "Style", "Country", "Brand"])

    df = df.reset_index(drop=True)

    skipped = total - len(df)
    print(f"Loaded {len(df)} rows ({skipped} skipped: unrated or missing fields)")

    return df
