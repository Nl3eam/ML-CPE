import pandas as pd

# The Kaggle export uses Title-Case headers ("Brand", "Stars", ...),
# some mirrors use snake_case ("brand", "stars", ...). Normalize both.
COLUMN_MAP = {
    "review_number": "Review #", "review #": "Review #", "review": "Review #",
    "brand": "Brand",
    "variety": "Variety",
    "style": "Style",
    "country": "Country",
    "stars": "Stars",
    "top_ten": "Top Ten", "top ten": "Top Ten",
}


def load_data(csv_path):
    """Load the raw Ramen Ratings csv and return a cleaned DataFrame.

    Rows with an unusable rating ("Unrated", blank, NaN) are dropped,
    since SVM classification needs a numeric target.
    """

    df = pd.read_csv(csv_path)

    # Normalize column names regardless of the source's casing
    df = df.rename(columns={c: COLUMN_MAP.get(c.strip().lower(), c) for c in df.columns})

    required = ["Brand", "Variety", "Style", "Country", "Stars"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = df[required].copy()

    # "Stars" is stored as text in the original export and contains a
    # handful of "Unrated" entries -> coerce to numeric and drop those
    df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Brand", "Variety", "Style", "Country", "Stars"])
    dropped = before - len(df)

    df["Brand"] = df["Brand"].astype(str).str.strip()
    df["Variety"] = df["Variety"].astype(str).str.strip()
    df["Style"] = df["Style"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} ramen reviews ({dropped} unrated/incomplete rows skipped)")

    return df


def group_rare_categories(series, top_n, other_label="Other"):
    """Collapse all but the top_n most frequent categories into 'Other'.

    Keeps one-hot encoding from exploding into hundreds of near-empty
    columns for high-cardinality fields like Brand and Country.
    """

    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), other_label)
