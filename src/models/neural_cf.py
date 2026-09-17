"""Neural Collaborative Filtering (NeuMF, He et al., 2017) trained on implicit feedback.

Architecture:

    user, item ─► GMF embeddings (d) ─► element-wise product ───────────────┐
                                                                            ├─► concat ─► Linear(1)
    user, item ─► MLP embeddings (d) ─► concat(2d) ─► [Linear ─► ReLU ─► Dropout] × L ─┘

Every rated movie is a positive interaction (label 1). For each positive,
``num_negatives`` movies the user never rated are sampled as negatives (label 0)
and the model is optimized with ``BCEWithLogitsLoss``. The output is a
preference score used to rank movies, not a rating on the 0.5–5 scale.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import NCF_CONFIG, RANDOM_SEED, NCFConfig
from src.evaluation.metrics import evaluate_ranking
from src.features.collaborative_features import IdEncoder, train_test_split_by_user

logger = logging.getLogger(__name__)


def set_torch_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NeuMF(nn.Module):
    """Neural Matrix Factorization: a GMF branch fused with an MLP branch.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        embedding_dim: Embedding size of each branch.
        hidden_layers: Output size of each hidden MLP layer.
        dropout: Dropout probability after each hidden layer.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = NCF_CONFIG.embedding_dim,
        hidden_layers: tuple[int, ...] = NCF_CONFIG.hidden_layers,
        dropout: float = NCF_CONFIG.dropout,
    ) -> None:
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_mlp = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        in_size = embedding_dim * 2
        for out_size in hidden_layers:
            layers += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = out_size
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_size + embedding_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.output.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return one logit per (user, item) pair."""
        gmf = self.user_gmf(user_ids) * self.item_gmf(item_ids)
        mlp = self.mlp(torch.cat([self.user_mlp(user_ids), self.item_mlp(item_ids)], dim=-1))
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


@dataclass
class TrainingHistory:
    """Per-epoch training loss and validation ranking quality."""

    train_loss: list[float] = field(default_factory=list)
    val_ndcg: list[float] = field(default_factory=list)
    val_recall: list[float] = field(default_factory=list)
    best_epoch: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": range(1, len(self.train_loss) + 1),
                "train_loss": self.train_loss,
                "val_ndcg@10": self.val_ndcg,
                "val_recall@10": self.val_recall,
            }
        )


class NCFRecommender:
    """Train and serve a :class:`NeuMF` model with negative sampling and early stopping.

    Args:
        user_encoder: Encoder for ``userId`` (defines embedding rows).
        item_encoder: Encoder for ``movieId`` (defines embedding rows).
        config: Training hyperparameters.
        device: ``"cuda"``, ``"cpu"`` or ``None`` to auto-detect.
        seed: Random seed.
    """

    def __init__(
        self,
        user_encoder: IdEncoder,
        item_encoder: IdEncoder,
        config: NCFConfig = NCF_CONFIG,
        device: str | None = None,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        self.model: NeuMF | None = None
        self.history = TrainingHistory()

    @property
    def n_users(self) -> int:
        return len(self.user_encoder)

    @property
    def n_items(self) -> int:
        return len(self.item_encoder)

    def _sample_epoch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        positive_keys: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one epoch of positives plus freshly sampled negatives."""
        n_neg = self.config.num_negatives
        neg_users = np.repeat(users, n_neg)
        neg_items = rng.integers(0, self.n_items, size=len(neg_users))
        # Re-draw any "negative" that is actually a positive for that user.
        clash = np.isin(neg_users * self.n_items + neg_items, positive_keys)
        while clash.any():
            neg_items[clash] = rng.integers(0, self.n_items, size=int(clash.sum()))
            clash = np.isin(neg_users * self.n_items + neg_items, positive_keys)

        all_users = torch.as_tensor(np.concatenate([users, neg_users]), device=self.device)
        all_items = torch.as_tensor(np.concatenate([items, neg_items]), device=self.device)
        labels = torch.cat([torch.ones(len(users)), torch.zeros(len(neg_users))]).to(self.device)
        return all_users, all_items, labels

    def fit(self, train_ratings: pd.DataFrame, verbose: bool = True) -> NCFRecommender:
        """Train NeuMF, early-stopping on validation NDCG@10.

        A ``config.val_size`` fraction of each user's training ratings is held out;
        after every epoch all movies are ranked for every user and NDCG@10 is computed
        against the held-out movies rated ``>= 4``. The best epoch's weights are kept.

        Args:
            train_ratings: Training ratings (``userId``, ``movieId``, ``rating``).
            verbose: Log progress every epoch.
        """
        cfg = self.config
        set_torch_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        fit_part, val_part = train_test_split_by_user(train_ratings, cfg.val_size, self.seed)
        users = self.user_encoder.transform(fit_part["userId"])
        items = self.item_encoder.transform(fit_part["movieId"])
        positive_keys = np.unique(users * self.n_items + items)

        self.model = NeuMF(
            self.n_users,
            self.n_items,
            embedding_dim=cfg.embedding_dim,
            hidden_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        self.history = TrainingHistory()
        best_ndcg, best_state, stale = -1.0, None, 0
        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()
            all_users, all_items, labels = self._sample_epoch(users, items, positive_keys, rng)
            order = torch.randperm(len(labels), device=self.device)
            total_loss = 0.0
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start : start + cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(all_users[batch], all_items[batch]), labels[batch])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            val = evaluate_ranking(
                self.score_users(), fit_part, val_part, self.user_encoder, self.item_encoder, (10,)
            )
            self.history.train_loss.append(total_loss / len(order))
            self.history.val_ndcg.append(val["NDCG@10"])
            self.history.val_recall.append(val["Recall@10"])
            if verbose:
                logger.info(
                    "Epoch %02d | loss %.4f | val NDCG@10 %.4f | val Recall@10 %.4f",
                    epoch,
                    total_loss / len(order),
                    val["NDCG@10"],
                    val["Recall@10"],
                )

            if val["NDCG@10"] > best_ndcg + 1e-4:
                best_ndcg, stale = val["NDCG@10"], 0
                best_state = copy.deepcopy(self.model.state_dict())
                self.history.best_epoch = epoch
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _check_fitted(self) -> NeuMF:
        if self.model is None:
            raise RuntimeError("Call fit() before predicting.")
        return self.model

    @torch.no_grad()
    def score_users(self, batch_users: int = 64) -> np.ndarray:
        """Preference probability for every (user, item) pair, ``[n_users × n_items]``."""
        model = self._check_fitted()
        was_training = model.training
        model.eval()
        all_items = torch.arange(self.n_items, device=self.device)
        scores = np.empty((self.n_users, self.n_items), dtype=np.float32)
        for start in range(0, self.n_users, batch_users):
            users = torch.arange(start, min(start + batch_users, self.n_users), device=self.device)
            logits = model(users.repeat_interleave(self.n_items), all_items.repeat(len(users)))
            scores[start : start + len(users)] = (
                torch.sigmoid(logits).view(len(users), self.n_items).cpu().numpy()
            )
        model.train(was_training)
        return scores

    def save(self, path: str) -> None:
        """Save model weights and config to ``path`` (``.pth``)."""
        model = self._check_fitted()
        torch.save({"state_dict": model.state_dict(), "config": self.config}, path)
