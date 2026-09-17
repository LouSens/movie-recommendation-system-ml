"""Smoke tests: every plot function renders and saves a figure from small inputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.utils import visualization as viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ratings(synthetic_ratings: pd.DataFrame) -> pd.DataFrame:
    return synthetic_ratings.assign(rating=synthetic_ratings["rating"].clip(0.5, 5.0))


@pytest.fixture
def movies() -> pd.DataFrame:
    genres = ["Action|Sci-Fi", "Comedy", "Drama|Romance", "(no genres listed)"]
    return pd.DataFrame(
        {
            "movieId": range(1, 41),
            "title": [f"Movie {i} (2000)" for i in range(1, 41)],
            "genres": [genres[i % 4] for i in range(40)],
        }
    )


def test_eda_plots_save_files(tmp_path, ratings, movies, raw_tags):
    # Make sure the 4.0 threshold line has a bar to anchor to.
    ratings = pd.concat([ratings, ratings.head(1).assign(rating=4.0, movieId=40, userId=99)])
    calls = {
        "01.png": lambda p: viz.plot_rating_distribution(ratings, save_path=p),
        "02.png": lambda p: viz.plot_genre_frequency(movies, save_path=p),
        "03.png": lambda p: viz.plot_user_activity(ratings, save_path=p),
        "04.png": lambda p: viz.plot_long_tail(ratings, save_path=p),
        "05.png": lambda p: viz.plot_ratings_per_year(ratings, save_path=p),
        "06.png": lambda p: viz.plot_top_tags(raw_tags, top_n=3, save_path=p),
        "07.png": lambda p: viz.plot_genre_ratings(movies, ratings, save_path=p),
        "08.png": lambda p: viz.plot_sparsity(ratings, size=10, save_path=p),
    }
    for name, draw in calls.items():
        fig = draw(tmp_path / name)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / name).stat().st_size > 0


def test_modeling_and_evaluation_plots(tmp_path):
    titles = ["A", "B", "C"]
    similarity = pd.DataFrame(np.eye(3), index=titles, columns=titles)
    viz.plot_similarity_heatmap(similarity, save_path=tmp_path / "09.png")

    cv_table = pd.DataFrame(
        {
            "param_n_factors": [100, 50],
            "param_reg_all": [0.1, 0.02],
            "mean_test_rmse": [0.85, 0.88],
            "std_test_rmse": [0.005, 0.004],
        }
    )
    viz.plot_svd_grid_search(cv_table, save_path=tmp_path / "10.png")

    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_ndcg@10": [0.1, 0.12, 0.11],
            "val_recall@10": [0.1, 0.13, 0.12],
        }
    )
    viz.plot_training_history(history, save_path=tmp_path / "11.png")

    results = pd.DataFrame(
        {"Precision@10": [0.1, 0.2], "NDCG@10": [0.15, 0.25], "Coverage@10": [0.5, 0.1]},
        index=["m1", "m2"],
    )
    viz.plot_metric_comparison(results, save_path=tmp_path / "12.png")

    y_true = np.array([1.0, 3.0, 4.0, 5.0, 4.0])
    viz.plot_svd_errors(y_true, y_true + 0.3, save_path=tmp_path / "13.png")

    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{i:02d}.png" for i in range(9, 14)]
