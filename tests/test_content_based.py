"""Tests for TF-IDF features and the content-based recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TfidfConfig
from src.features.collaborative_features import IdEncoder
from src.features.content_features import build_content_matrix
from src.models.content_based import ContentBasedRecommender

SMALL = TfidfConfig(ngram_range=(1, 1), min_df=1, max_features=100, genre_weight=0.5)


def test_content_matrix_rows_have_unit_norm(movies_content):
    features = build_content_matrix(movies_content, SMALL)
    norms = np.sqrt(features.matrix.multiply(features.matrix).sum(axis=1)).A1
    tagged = movies_content["tags"] != ""
    # Tagged movies: both blocks are present → norm 1; untagged: only sqrt(w_genre).
    assert np.allclose(norms[tagged], 1.0)
    assert np.allclose(norms[~tagged], np.sqrt(SMALL.genre_weight))


def test_invalid_genre_weight_raises(movies_content):
    with pytest.raises(ValueError):
        build_content_matrix(movies_content, TfidfConfig(genre_weight=1.5))


def test_recommend_similar_prefers_shared_genres_and_tags(movies_content):
    cbf = ContentBasedRecommender(movies_content, config=SMALL).fit()
    recs = cbf.recommend_similar("Toy Story (1995)", top_n=3)
    assert recs.loc[0, "title"] == "Toy Story 2 (1999)"
    assert "Toy Story (1995)" not in set(recs["title"])
    assert recs["similarity"].is_monotonic_decreasing


def test_recommend_similar_partial_title_and_unknown(movies_content):
    cbf = ContentBasedRecommender(movies_content, config=SMALL).fit()
    assert cbf.find_title("heat") == "Heat (1995)"
    with pytest.raises(ValueError):
        cbf.recommend_similar("Nonexistent Movie")


def test_recommend_similar_respects_candidates(movies_content):
    cbf = ContentBasedRecommender(movies_content, config=SMALL).fit()
    recs = cbf.recommend_similar("Toy Story", top_n=5, candidate_ids={3, 4})
    assert set(recs["movieId"]) <= {3, 4}


def test_unfitted_model_raises(movies_content):
    with pytest.raises(RuntimeError):
        ContentBasedRecommender(movies_content).recommend_similar("Heat")


def test_popularity_breaks_ties(movies_content):
    popularity = pd.Series({3: 1, 4: 100})
    cbf = ContentBasedRecommender(movies_content, config=SMALL, popularity=popularity).fit()
    recs = cbf.recommend_similar("Alien", top_n=2)
    # Alien shares no genre or tag with any movie, so every score is 0 and popularity decides.
    assert (recs["similarity"] == 0).all()
    assert recs["movieId"].tolist() == [4, 3]


def test_score_users_ranks_similar_content_higher(movies_content):
    cbf = ContentBasedRecommender(movies_content, config=SMALL).fit()
    train = pd.DataFrame({"userId": [1, 1], "movieId": [1, 4], "rating": [5.0, 1.0]})
    users, items = IdEncoder.fit([1]), IdEncoder.fit([1, 2, 3, 4, 5])
    scores = cbf.score_users(train, users, items)
    assert scores.shape == (1, 5)
    # Liked Toy Story, disliked Casino → Toy Story 2 should beat Heat (crime).
    assert scores[0, items.to_index[2]] > scores[0, items.to_index[3]]
