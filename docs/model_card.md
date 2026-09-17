# Model Card — Movie Recommendation System

## Overview

| | |
|---|---|
| **Task** | Top-N movie recommendation; rating prediction |
| **Models** | Content-Based (weighted TF-IDF), SVD (Surprise), NeuMF (PyTorch) |
| **Primary model** | NeuMF, for personalized top-N |
| **Author** | David Kurniawan |
| **Version** | 1.0.0 |
| **Framework versions (tested)** | Python 3.12.4, PyTorch 2.6.0+cu124, scikit-learn 1.7.2, scikit-surprise 1.1.5, pandas 2.2.2, NumPy 1.26.4 |

## Intended use

- **In scope:** offline research and education on recommender systems; a "because you watched X" feature (CBF); a "recommended for you" row (NeuMF); showing estimated star ratings (SVD).
- **Out of scope:** production use without retraining on in-domain data, fairness auditing, and live A/B testing. Scores are not calibrated probabilities of watching.

## Training data

MovieLens Latest Small (GroupLens, snapshot of September 2018): 100,836 ratings (0.5–5.0 stars), 3,683 tags, 9,742 movies, 610 users, collected 1996–2018.

After preparation:

| | Value |
|---|---|
| Cold-start filter | users ≥ 20 ratings, movies ≥ 5 ratings |
| Ratings / users / movies | 90,109 / 602 / 3,643 |
| Train / test | 72,300 / 17,809 (80:20 per user, seed 42) |
| NeuMF validation | 10% per-user hold-out of the training set |

## Models and hyperparameters

| Model | Configuration |
|---|---|
| Content-Based | Genre TF-IDF (20 tokens) ⊕ tag TF-IDF (1–2 grams, `min_df=2`, ≤ 5,000 → 1,114 terms); `genre_weight = 0.5`; profile weights = mean-centred ratings |
| SVD | Grid over `n_factors {50,100}`, `n_epochs {20,30}`, `lr_all {0.005,0.01}`, `reg_all {0.02,0.1}`, 5-fold CV. **Best: 100 / 30 / 0.01 / 0.1** (CV RMSE 0.8487) |
| NeuMF | Embedding 64 (GMF + MLP), MLP 256-128-64-32, dropout 0.2, 4 negatives per positive, Adam lr 1e-3, batch 2048, max 30 epochs, patience 5. **Best epoch 8** (val NDCG@10 0.1178), 619,713 parameters |
| Popularity | Count of training ratings ≥ 4.0 per movie |

## Evaluation results (test set)

Relevant = rating ≥ 4.0; 592 users evaluated; 3,643 candidate movies; items rated in training are excluded.

| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Coverage@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Content-Based | 0.0328 | 0.0126 | 0.0391 | 0.0279 | 0.0222 | 0.0368 | 0.4661 | – | – |
| SVD | 0.0155 | 0.0058 | 0.0166 | 0.0152 | 0.0125 | 0.0182 | 0.0571 | **0.8410** | **0.6454** |
| **NeuMF** | **0.1970** | **0.0963** | **0.2174** | **0.1667** | **0.1604** | **0.2173** | 0.1691 | – | – |
| Popularity | 0.1537 | 0.0668 | 0.1747 | 0.1248 | 0.1014 | 0.1653 | 0.0135 | – | – |
| Global mean | – | – | – | – | – | – | – | 1.0281 | 0.8182 |

## Limitations

1. **Small, old, US-centric dataset.** There are 610 users and ratings stop in 2018, so the results may not transfer to other platforms, regions or recent catalogues.
2. **Random split, not temporal.** The test set can contain ratings made *before* some training ratings, which is optimistic compared with predicting the future.
3. **Cold start.** SVD and NeuMF can't score unseen users or movies. The filter removed 62% of movies from the collaborative candidate set. CBF covers new movies only through genres/tags, and 84% of movies have no tags.
4. **Missing-not-at-random feedback.** Users mostly rate movies they chose to watch, which favours popular titles in both training and evaluation. That's why the popularity baseline is strong.
5. **Single run.** Metrics come from one seed/split; no confidence intervals are reported. GPU kernels can add tiny run-to-run variation on other hardware.
6. **NeuMF vs MF.** Well-tuned dot-product MF trained for ranking (e.g. implicit ALS / BPR) can match NCF (Rendle et al., 2020). Here SVD was tuned for RMSE, not ranking.

## Ethical considerations

- **Popularity bias / filter bubbles:** NeuMF's top-10 picks have a median of 107 training ratings, so they lean towards well-known titles. CBF is the most diverse (46.6% coverage). A production system should monitor diversity and exposure.
- **Privacy:** MovieLens user ids are anonymized. Real deployments must handle viewing history as personal data.
- **Tag noise:** free-text tags may contain subjective or offensive labels. Only normalization is applied, with no content moderation.

## How to reproduce

```bash
pip install -r requirements.txt
python -m src.pipeline            # CLI, writes outputs/results/pipeline_evaluation_metrics.csv
jupyter nbconvert --to notebook --execute notebooks/sistem_rekomendasi_film.ipynb
```
