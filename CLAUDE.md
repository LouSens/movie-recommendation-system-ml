# CLAUDE.md — Movie Recommendation System
> **Context file for Claude Code** — Read this before generating or modifying any code in this repository.

---

## 🎯 Project Purpose

Production-ready **Movie Recommendation System** with a self-contained notebook, an Indonesian report and a submission package. Implements two recommendation paradigms:

1. **Content-Based Filtering (CBF)** — TF-IDF on genres and tags (separate, weighted blocks) + weighted cosine similarity
2. **Collaborative Filtering (CF)** — SVD (Surprise library, rating prediction) + Neural Collaborative Filtering (NeuMF, PyTorch/CUDA, implicit feedback for top-N ranking)

---

## 📌 Project Context

| Key | Value |
|-----|-------|
| **Author** | David Kurniawan |
| **Submission Type** | Sistem Rekomendasi (Recommendation System) |
| **Dataset** | MovieLens Latest Small (`ml-latest-small`) |
| **Dataset Source** | https://grouplens.org/datasets/movielens/latest/ |
| **Repository** | https://github.com/LouSens/movie-recommendation-system-ml |
| **Hardware** | RTX 4050 (6GB VRAM), 16GB RAM, i9-13650HX |
| **Python Version** | 3.10+ (tested 3.12.4) |
| **CUDA Version** | 12.1+ (tested torch 2.6.0+cu124) |
| **Primary Language** | Python |
| **Docs language** | English (README, docs/, code); **Indonesian** for the submission report and notebook text cells |

---

## 📂 Repository Structure

```
movie-recommendation-system-ml/
│
├── CLAUDE.md                          # ← This file (Claude Code context)
├── README.md                          # GitHub project documentation
├── laporan_proyek_machine_learning.md # Project report (Indonesian)
├── sistem_rekomendasi_film.py         # Exported .py (submission)
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # pytest + ruff config
├── .gitignore
│
├── notebooks/
│   └── sistem_rekomendasi_film.ipynb  # Main notebook (self-contained, does NOT import src/)
│
├── src/
│   ├── config.py                      # Global config, paths, hyperparameters
│   ├── pipeline.py                    # End-to-end CLI: python -m src.pipeline
│   ├── data/
│   │   ├── downloader.py              # Dataset download utility
│   │   ├── loader.py                  # Raw data loading + validation
│   │   └── preprocessor.py            # Cleaning, content docs, cold-start filter
│   ├── features/
│   │   ├── content_features.py        # Weighted genre ⊕ tag TF-IDF matrix
│   │   └── collaborative_features.py  # IdEncoder, per-user split, user-item matrix
│   ├── models/
│   │   ├── content_based.py           # CBF: weighted cosine similarity engine
│   │   ├── svd_cf.py                  # CF: Surprise SVD wrapper + vectorized scoring
│   │   ├── neural_cf.py               # CF: PyTorch NeuMF (not re-exported from src.models)
│   │   └── baseline.py                # Popularity baseline
│   ├── evaluation/
│   │   └── metrics.py                 # Precision@K, Recall@K, NDCG@K, coverage, RMSE, MAE
│   └── utils/
│       ├── helpers.py                 # Seeding, pickling, top_n_for_user
│       └── visualization.py           # All 13 figures; embedded verbatim in the notebook
│
├── scripts/build_submission.py        # Builds dist/submission_movie_recommendation.zip
├── docs/                              # architecture.md, model_card.md, submission_checklist.md
│
├── data/raw/, data/processed/         # gitignored (except .gitkeep)
├── models/saved/                      # gitignored: svd_model.pkl, ncf_model.pth
├── outputs/figures/                   # 13 figures (committed; used by the report). Notebook and CLI write identical images here
├── outputs/results/                   # evaluation_metrics.csv, svd_grid_search.csv
│
└── tests/                             # 36 tests on synthetic fixtures (conftest.py)
    ├── test_loader.py
    ├── test_content_based.py
    ├── test_collaborative.py
    ├── test_metrics.py
    └── test_visualization.py
```

---

## 🔧 Environment Setup

```bash
git clone https://github.com/LouSens/movie-recommendation-system-ml.git
cd movie-recommendation-system-ml
conda create -n recsys python=3.12 -y && conda activate recsys
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -r requirements.txt
python -m src.data.downloader                                          # → data/raw/
```

> **Jupyter note:** make sure the notebook kernel's interpreter is the environment where the requirements are installed (`jupyter kernelspec list` → check `kernel.json`). A kernel pointing at another env fails with `No module named 'torch'`.

---

## 🚀 ML Pipeline Overview

```
[1. Data Loading & Validation]        ← src/data/loader.py
[2. Preprocessing]                    ← src/data/preprocessor.py
    - (no genres listed) → Unknown, genre_list, year
    - ratings: range check, dedupe (user, movie), timestamp → datetime
    - tags: normalize, dedupe (movie, tag), aggregate per movie
    - genre_tokens (Sci-Fi → scifi) + tags per movie
    - iterative cold-start filter: users ≥ 20, movies ≥ 5 ratings
[3. EDA]                              ← src/utils/visualization.py (13 figures in the notebook)
[4. Features]                         ← src/features/
    - CBF: TF-IDF(genres) ⊕ TF-IDF(tags, 1–2 gram, min_df 2), L2-normed, × √w / √(1−w), w = 0.5
    - CF: 80:20 split per user → IdEncoder → sparse user-item matrix (602 × 3,643)
[5. Models]                           ← src/models/
    - CBF: item-to-item (linear_kernel = weighted cosine) + mean-centred user profile
    - SVD: GridSearchCV 5-fold on train; n_factors [50,100], n_epochs [20,30],
           lr_all [0.005,0.01], reg_all [0.02,0.1] → best 100/30/0.01/0.1
    - NeuMF: GMF + MLP [256→128→64→32], emb 64, dropout 0.2, BCEWithLogitsLoss,
             4 negatives/positive resampled per epoch, Adam 1e-3, batch 2048,
             early stopping on validation NDCG@10 (10% per-user hold-out, patience 5)
[6. Evaluation]                       ← src/evaluation/metrics.py
    - Every model → score matrix [n_users × n_items]; train items masked; relevant = test rating ≥ 4
    - Precision/Recall/NDCG@{5,10}, Coverage@10 for all; RMSE/MAE for SVD; popularity baseline
[7. Top-N]                            ← ContentBasedRecommender.recommend_similar, SVDRecommender.recommend,
                                         utils.helpers.top_n_for_user
```

---

## 📊 Current Results (test set, seed 42)

| Model | P@10 | R@10 | NDCG@10 | Cov@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Content-Based | 0.0279 | 0.0222 | 0.0368 | 0.4661 | – | – |
| SVD | 0.0152 | 0.0125 | 0.0182 | 0.0571 | 0.8410 | 0.6454 |
| **NeuMF** | **0.1667** | **0.1604** | **0.2173** | 0.1691 | – | – |
| Popularity | 0.1248 | 0.1014 | 0.1653 | 0.0135 | – | – |

If you change modeling code, **re-execute the notebook, re-export the .py, and update the numbers** in the report, README, docs/model_card.md and this table.

---

## 💡 Key Architectural Decisions

### Separate, weighted genre and tag TF-IDF blocks (not one document)
Only 16% of movies have tags. A single concatenated document let tag-heavy movies drown out genres (*Toy Story 2* missing from *Toy Story*'s neighbours). Separate blocks give `sim = w·cos_genre + (1−w)·cos_tag` and raised CBF NDCG@10 from 0.015 to 0.037.

### NeuMF trained on implicit feedback (BCE + negative sampling)
A rating-regression NCF (MSE) reached RMSE 0.846 but NDCG@10 of only 0.011, below the popularity baseline. The product goal is top-N ranking, so NCF optimizes for it; SVD covers rating prediction.

### SVD via Surprise (not from scratch)
Robust, tested implementation with built-in CV. Scores for all pairs are computed from `pu, qi, bu, bi` in one vectorized call and **not clipped** for ranking (clipping creates ties at 5.0).

### Popularity baseline is always reported
MovieLens popularity is strong; SVD and CBF are below it on ranking metrics, NeuMF beats it. Keep it as a sanity floor.

### Self-contained notebook
The submission zip only contains the notebook, the .py and the report (+ figures), so the notebook must not import `src/`. Model/data logic is duplicated deliberately — keep both in sync.

### Notebook and .py are standalone
The notebook never imports `src/`, `visualization.py` or any other repo file: it defines its own copies of every function and downloads the data itself. It must run in an empty folder and still produce all 13 figures. The submission `.py` must contain **exactly** the notebook's code cells (only exported via `jupyter nbconvert --to script`, never hand-edited). `scripts/build_submission.py` enforces both and refuses to build otherwise.

### Plotting code is kept in sync with `src/utils/visualization.py`
The notebook holds its own **verbatim copies** of these functions (no import), grouped by section (`SHARED_HELPERS` + `EDA_PLOTS` at the start of EDA; `plot_similarity_heatmap`, `plot_svd_grid_search`, `plot_training_history` right before first use in Modeling; `EVALUATION_PLOTS` in Evaluation). After editing the module (including `ruff format`), regenerate and re-execute the notebook, then check each function's source still appears verbatim in the notebook. Chart labels are Indonesian because the figures go into the Indonesian report.

### One figures folder
The CLI pipeline saves to `outputs/figures/` too, with the same functions, raw inputs, model names (`CF — SVD`, `CF — NeuMF`) and model order as the notebook, so it overwrites with identical images. Keep these in sync if you rename models in either place. Running the pipeline is optional.

### Cold-start handling
CBF handles new movies; CF uses the iterative ≥ 20 / ≥ 5 filter. A hybrid is on the roadmap.

---

## 🎨 Code Style

- **PEP 8**, `ruff check` + `ruff format` (config in pyproject.toml)
- Docstrings: **Google style** for all public functions
- Type hints on all function signatures
- Max line length: **100 characters**
- Imports grouped (stdlib → third-party → local), sorted (ruff `I`)

---

## 🧪 Commands

```bash
pytest                                        # 36 tests (~15 s)
ruff check src tests scripts && ruff format --check src tests scripts
python -m src.pipeline [--skip-tuning] [--skip-ncf]   # optional; the notebook alone covers the submission
jupyter nbconvert --to notebook --execute --inplace notebooks/sistem_rekomendasi_film.ipynb
jupyter nbconvert --to script notebooks/sistem_rekomendasi_film.ipynb --output-dir .
python scripts/build_submission.py            # checks: notebook executed, no repo imports, .py == notebook code, images exist
```

---

## ⚠️ Common Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| `No module named 'torch'` in Jupyter | Kernel points to another env | Use a kernel for the env with requirements installed |
| `CUDA out of memory` | Batch too large | Reduce `NCFConfig.batch_size` or `score_users(batch_users=...)` |
| `Surprise ImportError` | Missing scikit-surprise | `pip install scikit-surprise` (not `surprise`) |
| `ValueError: Movie '...' not found` | Title not in catalogue | Use partial title (`find_title`) or fall back to popular movies |
| Metrics slightly differ on other GPUs | Non-deterministic CUDA kernels | Expect ± small variation; CPU runs are deterministic |

---

## 📈 Performance Benchmarks (Local RTX 4050)

| Stage | Duration |
|-------|-----------------|
| TF-IDF + CBF scores | ~0.2 s |
| SVD GridSearchCV (16 × 5-fold) | ~7–12 s |
| NeuMF training (early stop at epoch 13) | ~15 s |
| Full pipeline CLI | ~40 s |
| Full notebook execution | ~2 min |

---

## 🗒️ TODO / Roadmap

- [x] Dataset download script
- [x] EDA pipeline
- [x] Content-Based Filtering
- [x] SVD Collaborative Filtering
- [x] Neural CF (PyTorch CUDA)
- [ ] Hybrid model (CBF + CF weighted blend)
- [ ] Temporal evaluation split
- [ ] FastAPI REST deployment
- [ ] Streamlit demo UI
- [ ] Docker containerization
