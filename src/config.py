"""Global configuration: filesystem paths, random seeds and hyperparameters.

Every module imports its constants from here so that the notebook, the CLI
pipeline and the tests share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

DATASET_URL: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_NAME: str = "ml-latest-small"
DATASET_FILES: tuple[str, ...] = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
MIN_USER_RATINGS: int = 20
MIN_MOVIE_RATINGS: int = 5
TEST_SIZE: float = 0.2
RATING_SCALE: tuple[float, float] = (0.5, 5.0)
RELEVANCE_THRESHOLD: float = 4.0


@dataclass(frozen=True)
class TfidfConfig:
    """Hyperparameters for the content-based TF-IDF features (n-gram settings apply to tags)."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int = 5000
    genre_weight: float = 0.5  # sim = w·cos(genres) + (1 − w)·cos(tags)


@dataclass(frozen=True)
class SVDConfig:
    """Search space for the Surprise SVD grid search."""

    param_grid: dict[str, list] = field(
        default_factory=lambda: {
            "n_factors": [50, 100],
            "n_epochs": [20, 30],
            "lr_all": [0.005, 0.01],
            "reg_all": [0.02, 0.1],
        }
    )
    cv_folds: int = 5


@dataclass(frozen=True)
class NCFConfig:
    """Hyperparameters for the PyTorch NeuMF (Neural Collaborative Filtering) model."""

    embedding_dim: int = 64
    hidden_layers: tuple[int, ...] = (256, 128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    num_negatives: int = 4
    max_epochs: int = 30
    patience: int = 5
    val_size: float = 0.1


TFIDF = TfidfConfig()
SVD_CONFIG = SVDConfig()
NCF_CONFIG = NCFConfig()

TOP_K_VALUES: tuple[int, ...] = (5, 10)
DEFAULT_TOP_N: int = 10
