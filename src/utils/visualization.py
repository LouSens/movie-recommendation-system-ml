"""EDA and evaluation plots. Every function returns the figure and optionally saves it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = "viridis"


def _finish(fig: plt.Figure, save_path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_rating_distribution(ratings: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Bar chart of how often each rating value (0.5–5.0) is given."""
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = ratings["rating"].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3b7dd8", ax=ax)
    ax.axvline(
        x=list(counts.index).index(4.0) - 0.5,
        color="#d64545",
        linestyle="--",
        label="relevant ≥ 4.0",
    )
    ax.set(title="Rating distribution", xlabel="Rating", ylabel="Number of ratings")
    ax.legend()
    return _finish(fig, save_path)


def plot_genre_frequency(movies: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Horizontal bar chart of the number of movies per genre."""
    counts = movies["genre_list"].explode().value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=counts.values, y=counts.index, hue=counts.index, palette=PALETTE, legend=False, ax=ax
    )
    ax.set(title="Number of movies per genre", xlabel="Movies", ylabel="")
    return _finish(fig, save_path)


def plot_user_activity(ratings: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Histogram (log x-axis) of the number of ratings per user."""
    per_user = ratings.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40)
    ax.hist(per_user, bins=bins, color="#2a9d8f", edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(
        per_user.median(),
        color="#d64545",
        linestyle="--",
        label=f"median = {per_user.median():.0f}",
    )
    ax.set(title="Ratings per user", xlabel="Ratings per user (log scale)", ylabel="Users")
    ax.legend()
    return _finish(fig, save_path)


def plot_long_tail(ratings: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Movie popularity curve showing the long-tail of rating counts."""
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False).to_numpy()
    cumulative = per_movie.cumsum() / per_movie.sum()
    head = int(np.searchsorted(cumulative, 0.5)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(per_movie) + 1), per_movie, color="#264653")
    ax.fill_between(
        np.arange(1, head + 1),
        per_movie[:head],
        color="#e9c46a",
        alpha=0.6,
        label=f"top {head} movies = 50% of ratings",
    )
    ax.set_yscale("log")
    ax.set(title="Movie popularity (long tail)", xlabel="Movie rank", ylabel="Ratings (log scale)")
    ax.legend()
    return _finish(fig, save_path)


def plot_ratings_per_year(ratings: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Number of ratings and mean rating per calendar year."""
    yearly = ratings.groupby(ratings["datetime"].dt.year)["rating"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(yearly.index, yearly["size"], color="#8ab17d")
    ax.set(title="Ratings per year", xlabel="Year", ylabel="Number of ratings")
    ax2 = ax.twinx()
    ax2.plot(yearly.index, yearly["mean"], color="#d64545", marker="o")
    ax2.set_ylabel("Mean rating", color="#d64545")
    return _finish(fig, save_path)


def plot_top_tags(tags: pd.DataFrame, top_n: int = 20, save_path: Path | None = None) -> plt.Figure:
    """Most frequent normalized tags."""
    counts = tags["tag_clean"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=counts.values, y=counts.index, hue=counts.index, palette="mako", legend=False, ax=ax
    )
    ax.set(title=f"Top {top_n} user tags", xlabel="Movies tagged", ylabel="")
    return _finish(fig, save_path)


def plot_genre_ratings(
    movies: pd.DataFrame, ratings: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Box plot of per-movie mean rating by genre (movies with ≥ 5 ratings)."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "size"]).reset_index()
    stats = stats[stats["size"] >= 5].merge(movies[["movieId", "genre_list"]], on="movieId")
    exploded = stats.explode("genre_list")
    order = exploded.groupby("genre_list")["mean"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=exploded, x="mean", y="genre_list", order=order, color="#90be6d", fliersize=2, ax=ax
    )
    ax.set(
        title="Mean movie rating by genre (movies with ≥ 5 ratings)",
        xlabel="Mean rating",
        ylabel="",
    )
    return _finish(fig, save_path)


def plot_sparsity(matrix: np.ndarray, save_path: Path | None = None, size: int = 100) -> plt.Figure:
    """Show which cells of the densest ``size × size`` corner of a user–item matrix are filled."""
    dense = np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix)
    users = np.argsort(-(dense > 0).sum(axis=1))[:size]
    items = np.argsort(-(dense > 0).sum(axis=0))[:size]
    block = dense[np.ix_(users, items)] > 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(block, cmap="Greys", aspect="auto", interpolation="nearest")
    fill = (dense > 0).mean()
    ax.set(
        title=f"User–item matrix: top {size} users × top {size} movies\n"
        f"(overall density {fill:.2%})",
        xlabel="Movies",
        ylabel="Users",
    )
    return _finish(fig, save_path)


def plot_training_history(history: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Training loss and validation NDCG@10 / Recall@10 per epoch for Neural CF."""
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", color="#264653")
    ax_loss.set(title="Training loss (BCE)", xlabel="Epoch", ylabel="Loss")

    ax_val.plot(history["epoch"], history["val_ndcg@10"], marker="o", label="NDCG@10")
    ax_val.plot(history["epoch"], history["val_recall@10"], marker="o", label="Recall@10")
    best = history.loc[history["val_ndcg@10"].idxmax()]
    ax_val.axvline(
        best["epoch"], color="#d64545", linestyle="--", label=f"best epoch {int(best['epoch'])}"
    )
    ax_val.set(title="Validation ranking quality", xlabel="Epoch", ylabel="Score")
    ax_val.legend()
    return _finish(fig, save_path)


def plot_metric_comparison(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Grouped bar chart of ranking metrics per model.

    Args:
        results: DataFrame indexed by model name with metric columns such as ``Precision@10``.
    """
    metrics = [c for c in results.columns if "@" in c]
    long = results[metrics].reset_index(names="model").melt(id_vars="model", var_name="metric")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="metric", y="value", hue="model", palette="Set2", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    ax.set(title="Top-K ranking metrics on the test set", xlabel="", ylabel="Score")
    ax.set_ylim(0, long["value"].max() * 1.12)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    return _finish(fig, save_path)


def plot_rating_errors(
    y_true: np.ndarray, predictions: dict[str, np.ndarray], save_path: Path | None = None
) -> plt.Figure:
    """Distribution of prediction errors (predicted − actual) for each rating model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, preds in predictions.items():
        sns.kdeplot(np.asarray(preds) - np.asarray(y_true), ax=ax, label=name, fill=True, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Rating prediction error on the test set", xlabel="Predicted − actual rating")
    ax.legend()
    return _finish(fig, save_path)
