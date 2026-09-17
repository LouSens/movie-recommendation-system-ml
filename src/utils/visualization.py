"""EDA, modeling and evaluation plots shared by the notebook and the CLI pipeline.

This module is the single source of truth for every figure. The submission notebook
embeds these functions verbatim (it must stay self-contained), grouped by the section
that uses them: EDA, Modeling and Evaluation.

Chart labels are in Indonesian because the figures are embedded in the Indonesian
submission report. Each function draws one figure, saves it when ``save_path`` is
given and returns the :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ACCENT = "#d64545"


def _finish(fig: plt.Figure, save_path: Path | str | None) -> plt.Figure:
    """Tighten the layout and save ``fig`` to ``save_path`` (if given)."""
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def _genre_lists(genres: pd.Series) -> pd.Series:
    """Split ``Action|Comedy`` strings into lists."""
    return genres.fillna("").str.split("|")


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def plot_rating_distribution(
    ratings: pd.DataFrame, threshold: float = 4.0, save_path: Path | str | None = None
) -> plt.Figure:
    """Bar chart of how often each rating value (0.5–5.0) is given."""
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3b7dd8", ax=ax)
    ax.axvline(
        list(counts.index).index(threshold) - 0.5,
        color=ACCENT,
        ls="--",
        label=f"relevan (≥ {threshold})",
    )
    ax.set(title="Distribusi rating", xlabel="Rating", ylabel="Jumlah rating")
    ax.legend()
    return _finish(fig, save_path)


def plot_genre_frequency(movies: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Horizontal bar chart of the number of movies per genre."""
    counts = _genre_lists(movies["genres"]).explode().value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=False, ax=ax
    )
    ax.set(title="Jumlah film per genre", xlabel="Jumlah film", ylabel="")
    return _finish(fig, save_path)


def plot_user_activity(ratings: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Histogram (log x-axis) of the number of ratings per user."""
    per_user = ratings.groupby("userId").size()
    bins = np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_user, bins=bins, color="#2a9d8f", edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(per_user.median(), color=ACCENT, ls="--", label=f"median = {per_user.median():.0f}")
    ax.set(
        title="Jumlah rating per user", xlabel="Rating per user (skala log)", ylabel="Jumlah user"
    )
    ax.legend()
    return _finish(fig, save_path)


def plot_long_tail(ratings: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Movie popularity curve highlighting the head that receives 50% of all ratings."""
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False).to_numpy()
    cumulative = per_movie.cumsum() / per_movie.sum()
    head = int(np.searchsorted(cumulative, 0.5)) + 1
    ranks = np.arange(1, len(per_movie) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ranks, per_movie, color="#264653")
    ax.fill_between(
        ranks[:head],
        per_movie[:head],
        color="#e9c46a",
        alpha=0.6,
        label=f"{head} film teratas = 50% rating",
    )
    ax.set_yscale("log")
    ax.set(
        title="Popularitas film (long tail)",
        xlabel="Peringkat film",
        ylabel="Jumlah rating (skala log)",
    )
    ax.legend()
    return _finish(fig, save_path)


def plot_ratings_per_year(ratings: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Number of ratings (bars) and mean rating (line) per calendar year."""
    years = pd.to_datetime(ratings["timestamp"], unit="s").dt.year
    yearly = ratings.groupby(years)["rating"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(yearly.index, yearly["size"], color="#8ab17d")
    ax.set(title="Jumlah rating per tahun", xlabel="Tahun", ylabel="Jumlah rating")
    ax2 = ax.twinx()
    ax2.plot(yearly.index, yearly["mean"], color=ACCENT, marker="o")
    ax2.set_ylabel("Rata-rata rating", color=ACCENT)
    ax2.grid(False)
    return _finish(fig, save_path)


def plot_top_tags(
    tags: pd.DataFrame, top_n: int = 20, save_path: Path | str | None = None
) -> plt.Figure:
    """Most frequent tags (lower-cased and stripped)."""
    counts = tags["tag"].str.lower().str.strip().value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=counts.values, y=counts.index, hue=counts.index, palette="mako", legend=False, ax=ax
    )
    ax.set(title=f"{top_n} tag terpopuler", xlabel="Frekuensi", ylabel="")
    return _finish(fig, save_path)


def plot_genre_ratings(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 5,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Box plot of per-movie mean rating by genre, for movies with ``>= min_ratings``."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "size"]).reset_index()
    stats = stats[stats["size"] >= min_ratings].merge(movies[["movieId", "genres"]], on="movieId")
    exploded = stats.assign(genre=_genre_lists(stats["genres"])).explode("genre")
    order = exploded.groupby("genre")["mean"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=exploded, x="mean", y="genre", order=order, color="#90be6d", fliersize=2, ax=ax
    )
    ax.set(
        title=f"Rata-rata rating film per genre (film dengan ≥ {min_ratings} rating)",
        xlabel="Rata-rata rating film",
        ylabel="",
    )
    return _finish(fig, save_path)


def plot_sparsity(
    ratings: pd.DataFrame, size: int = 100, save_path: Path | str | None = None
) -> plt.Figure:
    """Filled cells of the ``size`` most active users × ``size`` most rated movies."""
    top_users = ratings["userId"].value_counts().index[:size]
    top_movies = ratings["movieId"].value_counts().index[:size]
    block = (
        ratings[ratings["userId"].isin(top_users) & ratings["movieId"].isin(top_movies)]
        .pivot_table(index="userId", columns="movieId", values="rating")
        .reindex(index=top_users, columns=top_movies)
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(block.notna(), cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set(
        title=f"Matriks user-item: {size} user teraktif × {size} film terpopuler\n"
        "(sel hitam = ada rating)",
        xlabel="Film",
        ylabel="User",
    )
    ax.grid(False)
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
def plot_similarity_heatmap(
    similarity: pd.DataFrame, save_path: Path | str | None = None
) -> plt.Figure:
    """Annotated heatmap of a square movie × movie similarity DataFrame."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(similarity, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, ax=ax)
    ax.set_title("Weighted cosine similarity antar film")
    return _finish(fig, save_path)


def plot_svd_grid_search(cv_table: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Mean ± std CV RMSE for every SVD hyperparameter combination (best on top).

    Args:
        cv_table: One row per combination with ``param_*``, ``mean_test_rmse`` and
            ``std_test_rmse`` columns, sorted from best to worst.
    """
    param_cols = [c for c in cv_table.columns if c.startswith("param_") and "random" not in c]

    def label(row: pd.Series) -> str:
        return ", ".join(f"{c.removeprefix('param_')}={row[c]:g}" for c in param_cols)

    labels = cv_table.apply(label, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        labels[::-1],
        cv_table["mean_test_rmse"][::-1],
        xerr=cv_table["std_test_rmse"][::-1],
        color="#3b7dd8",
    )
    ax.set_xlim(cv_table["mean_test_rmse"].min() - 0.02, cv_table["mean_test_rmse"].max() + 0.01)
    ax.set(title="SVD GridSearchCV — rata-rata RMSE 5-fold (lebih kecil lebih baik)", xlabel="RMSE")
    ax.tick_params(axis="y", labelsize=7)
    return _finish(fig, save_path)


def plot_training_history(history: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Training loss and validation NDCG@10 / Recall@10 per epoch for NeuMF."""
    best_epoch = int(history.loc[history["val_ndcg@10"].idxmax(), "epoch"])
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", color="#264653")
    ax_loss.set(title="Training loss (BCE)", xlabel="Epoch", ylabel="Loss")
    ax_val.plot(history["epoch"], history["val_ndcg@10"], marker="o", label="NDCG@10")
    ax_val.plot(history["epoch"], history["val_recall@10"], marker="o", label="Recall@10")
    ax_val.axvline(best_epoch, color=ACCENT, ls="--", label=f"epoch terbaik = {best_epoch}")
    ax_val.set(title="Validasi NeuMF", xlabel="Epoch", ylabel="Skor")
    ax_val.legend()
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def plot_metric_comparison(
    results: pd.DataFrame, save_path: Path | str | None = None
) -> plt.Figure:
    """Grouped bar chart of Precision / Recall / NDCG @K per model.

    Args:
        results: DataFrame indexed by model name; other columns (RMSE, coverage, ...)
            are ignored.
    """
    metrics = [c for c in results.columns if c.split("@")[0] in ("Precision", "Recall", "NDCG")]
    long = results[metrics].reset_index(names="model").melt(id_vars="model", var_name="metric")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=long, x="metric", y="value", hue="model", palette="Set2", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    ax.set(title="Perbandingan metrik ranking pada data uji", xlabel="", ylabel="Skor")
    ax.set_ylim(0, long["value"].max() * 1.12)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    return _finish(fig, save_path)


def plot_svd_errors(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | str | None = None
) -> plt.Figure:
    """Distribution of SVD prediction errors and MAE per actual rating value."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    errors = y_pred - y_true
    by_rating = (
        pd.DataFrame({"rating": y_true, "abs_err": np.abs(errors)})
        .groupby("rating")["abs_err"]
        .mean()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(errors, bins=40, kde=True, color="#3b7dd8", ax=ax1)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set(title="Distribusi galat prediksi SVD", xlabel="Prediksi − aktual")
    ax2.bar(by_rating.index.astype(str), by_rating.values, color="#e76f51")
    ax2.set(title="MAE SVD per nilai rating aktual", xlabel="Rating aktual", ylabel="MAE")
    return _finish(fig, save_path)


# Section grouping used to embed these functions in the notebook.
SHARED_HELPERS = ("ACCENT", "_finish", "_genre_lists")
EDA_PLOTS = (
    "plot_rating_distribution",
    "plot_genre_frequency",
    "plot_user_activity",
    "plot_long_tail",
    "plot_ratings_per_year",
    "plot_top_tags",
    "plot_genre_ratings",
    "plot_sparsity",
)
MODELING_PLOTS = ("plot_similarity_heatmap", "plot_svd_grid_search", "plot_training_history")
EVALUATION_PLOTS = ("plot_metric_comparison", "plot_svd_errors")
