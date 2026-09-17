"""End-to-end pipeline: download → prepare → train CBF / SVD / NCF → evaluate → save artefacts.

Usage:
    python -m src.pipeline                 # full run (SVD grid search + Neural CF)
    python -m src.pipeline --skip-tuning   # use default SVD params (fast)
    python -m src.pipeline --skip-ncf      # no PyTorch required
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RESULTS_DIR,
)
from src.data import (  # noqa: E402
    build_movie_content,
    clean_movies,
    clean_ratings,
    clean_tags,
    download_movielens,
    filter_cold_start,
    load_movielens,
)
from src.evaluation import catalog_coverage, evaluate_ranking, mae, rmse  # noqa: E402
from src.features import (  # noqa: E402
    IdEncoder,
    build_user_item_matrix,
    sparsity,
    train_test_split_by_user,
)
from src.models import ContentBasedRecommender, SVDRecommender, popularity_scores  # noqa: E402
from src.utils import save_pickle, set_seed, setup_logging, timer, top_n_for_user  # noqa: E402
from src.utils import visualization as viz  # noqa: E402

logger = logging.getLogger("pipeline")

# Model names and order match the notebook's evaluation table (legend labels and colors).
CBF, SVD, NEUMF, POPULARITY = (
    "Content-Based (TF-IDF)",
    "CF — SVD",
    "CF — NeuMF",
    "Popularity baseline",
)
MODEL_ORDER = (CBF, SVD, NEUMF, POPULARITY)


def run(skip_tuning: bool = False, skip_ncf: bool = False) -> pd.DataFrame:
    """Execute the full pipeline and return the evaluation table."""
    set_seed(RANDOM_SEED)
    viz.apply_theme()
    # Same functions, inputs and model names as the notebook, so these overwrite
    # outputs/figures with identical images.
    figures = FIGURES_DIR
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, figures, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Data loading ---------------------------------------------------------
    download_movielens(RAW_DATA_DIR)
    raw = load_movielens(RAW_DATA_DIR)
    logger.info("Raw tables:\n%s", raw.summary())

    # 2. Preparation ----------------------------------------------------------
    movies = clean_movies(raw.movies)
    ratings = clean_ratings(raw.ratings)
    tags = clean_tags(raw.tags)
    movies_content = build_movie_content(movies, tags)
    filtered = filter_cold_start(ratings)
    logger.info("Ratings after cold-start filter: %d (from %d)", len(filtered), len(ratings))

    # 3. EDA figures ----------------------------------------------------------
    # Same functions and inputs (raw tables) as the notebook's EDA section.
    viz.plot_rating_distribution(raw.ratings, save_path=figures / "01_rating_distribution.png")
    viz.plot_genre_frequency(raw.movies, save_path=figures / "02_genre_frequency.png")
    viz.plot_user_activity(raw.ratings, save_path=figures / "03_user_activity.png")
    viz.plot_long_tail(raw.ratings, save_path=figures / "04_long_tail.png")
    viz.plot_ratings_per_year(raw.ratings, save_path=figures / "05_ratings_per_year.png")
    viz.plot_top_tags(raw.tags, save_path=figures / "06_top_tags.png")
    viz.plot_genre_ratings(raw.movies, raw.ratings, save_path=figures / "07_genre_ratings.png")
    viz.plot_sparsity(raw.ratings, save_path=figures / "08_sparsity.png")
    plt.close("all")

    # 4. Split + encoders -----------------------------------------------------
    train, test = train_test_split_by_user(filtered)
    user_enc, item_enc = IdEncoder.fit(train["userId"]), IdEncoder.fit(train["movieId"])
    test = test[test["movieId"].isin(item_enc.to_index)].reset_index(drop=True)
    user_item = build_user_item_matrix(train, user_enc, item_enc)
    logger.info("Train %d | Test %d | Sparsity %.4f", len(train), len(test), sparsity(user_item))
    save_pickle(user_item, PROCESSED_DATA_DIR / "user_item_matrix.pkl")

    results: dict[str, dict[str, float]] = {}
    scores: dict[str, object] = {}

    # 5a. Content-based -------------------------------------------------------
    with timer("Content-based TF-IDF"):
        popularity = ratings["movieId"].value_counts()
        cbf = ContentBasedRecommender(movies_content, popularity=popularity).fit()
        scores[CBF] = cbf.score_users(train, user_enc, item_enc)
    save_pickle(cbf.tfidf_matrix, PROCESSED_DATA_DIR / "tfidf_matrix.pkl")
    sample_titles = [
        "Toy Story (1995)",
        "Toy Story 2 (1999)",
        "Monsters, Inc. (2001)",
        "Matrix, The (1999)",
        "Terminator 2: Judgment Day (1991)",
        "Godfather, The (1972)",
    ]
    sample_ids = [
        cbf.movies.loc[cbf.movies["title"] == t, "movieId"].iloc[0] for t in sample_titles
    ]
    heatmap = pd.DataFrame(
        cbf.similarity_matrix(sample_ids), index=sample_titles, columns=sample_titles
    )
    viz.plot_similarity_heatmap(heatmap, save_path=figures / "09_cbf_similarity_heatmap.png")
    similar = cbf.recommend_similar("Toy Story (1995)")
    logger.info("Similar to Toy Story:\n%s", similar.to_string())

    # 5b. SVD -----------------------------------------------------------------
    svd = SVDRecommender()
    if skip_tuning:
        svd.best_params = {"n_factors": 100, "n_epochs": 30, "lr_all": 0.01, "reg_all": 0.1}
    else:
        with timer("SVD grid search"):
            svd.tune(train)
        svd.cv_results.to_csv(RESULTS_DIR / "pipeline_svd_grid_search.csv", index=False)
        viz.plot_svd_grid_search(svd.cv_results, save_path=figures / "10_svd_grid_search.png")
    logger.info("SVD params: %s", svd.best_params)
    svd.fit(train)
    save_pickle(svd.model, MODELS_DIR / "svd_model.pkl")
    svd_test = svd.predict(test)
    results[SVD] = {"RMSE": rmse(test["rating"], svd_test), "MAE": mae(test["rating"], svd_test)}
    viz.plot_svd_errors(test["rating"], svd_test, save_path=figures / "13_svd_errors.png")
    scores[SVD] = svd.score_users(user_enc, item_enc)

    # 5c. Neural CF -----------------------------------------------------------
    if not skip_ncf:
        from src.models.neural_cf import NCFRecommender

        with timer("Neural CF training"):
            ncf = NCFRecommender(user_enc, item_enc).fit(train)
        ncf.save(str(MODELS_DIR / "ncf_model.pth"))
        viz.plot_training_history(
            ncf.history.to_frame(), save_path=figures / "11_ncf_learning_curve.png"
        )
        scores[NEUMF] = ncf.score_users()

    scores[POPULARITY] = popularity_scores(train, user_enc, item_enc)

    # 6. Evaluation -----------------------------------------------------------
    train_mask = user_item.toarray() > 0
    for name, matrix in scores.items():
        metrics = evaluate_ranking(matrix, train, test, user_enc, item_enc)
        metrics["Coverage@10"] = catalog_coverage(matrix, train_mask, k=10)
        results.setdefault(name, {}).update(metrics)

    table = pd.DataFrame(results).T.reindex([m for m in MODEL_ORDER if m in results])
    table.to_csv(RESULTS_DIR / "pipeline_evaluation_metrics.csv")
    viz.plot_metric_comparison(
        table.drop(columns=["Coverage@10"]), save_path=figures / "12_metric_comparison.png"
    )
    logger.info("Evaluation:\n%s", table.round(4).to_string())

    sample_user = int(user_enc.to_id[0])
    top = svd.recommend(sample_user, train, movies, item_enc)
    logger.info("SVD top-10 for user %d:\n%s", sample_user, top.to_string())
    if NEUMF in scores:
        top = top_n_for_user(
            sample_user, scores[NEUMF][0], train, movies, item_enc, score_name="score"
        )
        logger.info("Neural CF top-10 for user %d:\n%s", sample_user, top.to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the movie recommendation pipeline.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip SVD grid search.")
    parser.add_argument("--skip-ncf", action="store_true", help="Skip Neural CF training.")
    args = parser.parse_args()
    setup_logging()
    run(skip_tuning=args.skip_tuning, skip_ncf=args.skip_ncf)


if __name__ == "__main__":
    main()
