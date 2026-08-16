import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay

from preprocess import STARS_THRESHOLD, CLASSES


def plot_stars_distribution(df, save_path):
    """Histogram of the raw Stars rating, with the Good/Not-Good cutoff."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Stars"], bins=21, range=(0, 5), color="#e07b39", edgecolor="white")
    ax.axvline(STARS_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Good cutoff ({STARS_THRESHOLD})")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Ramen Star Rating Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_style_and_country(df, save_path, top_n=10):
    """Two-panel bar chart: review counts by Style, and avg Stars by top Country."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    style_counts = df["Style"].value_counts()
    axes[0].bar(style_counts.index, style_counts.values, color="#3f6f76")
    axes[0].set_title("Reviews by Style")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    top_countries = df["Country"].value_counts().nlargest(top_n).index
    avg_by_country = (
        df[df["Country"].isin(top_countries)]
        .groupby("Country")["Stars"].mean()
        .sort_values(ascending=False)
    )
    axes[1].bar(avg_by_country.index, avg_by_country.values, color="#a13d3d")
    axes[1].axhline(STARS_THRESHOLD, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"Avg Stars — Top {top_n} Countries by Review Count")
    axes[1].set_ylabel("Average Stars")
    axes[1].tick_params(axis="x", rotation=60)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pca_scatter(scaler, X, y, save_path):
    """Project the model's own PCA-reduced features down to 2D for a peek
    at how separable Good vs Not Good actually is."""

    # Reuse the already-fitted feature+scaler steps, only re-fit a 2D PCA
    # on top of them purely for visualization (not used by the model).
    pre_pca = scaler[:-1].transform(X)  # everything except the final PCA step
    coords = PCA(n_components=2, random_state=42).fit_transform(pre_pca)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label_value, label_name, color in zip([0, 1], CLASSES, ["#a13d3d", "#3f6f76"]):
        mask = y == label_value
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.5,
                   color=color, label=label_name)

    ax.set_title("2D PCA Projection of Ramen Features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curve(model, scaler, X_test, y_test, save_path):
    """ROC curve using the SVM's decision function on the held-out test set."""

    X_test_scaled = scaler.transform(X_test)
    scores = model.decision_function(X_test_scaled)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, scores, name="SVM (RBF)", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve — Good vs Not Good")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
