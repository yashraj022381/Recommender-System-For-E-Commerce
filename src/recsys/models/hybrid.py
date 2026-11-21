"""Hybrid recommender: weighted reciprocal-rank fusion (RRF) of several fitted models."""
import numpy as np
from scipy import sparse as sp

from recsys.data.loader import Dataset
from recsys.models.base import Recommender


def seen_matrix(df, n_users: int, n_items: int) -> sp.csr_matrix:
    """User x item matrix of everything a user already has (any label)."""
    return sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user_id"].to_numpy(), df["item_id"].to_numpy())),
        shape=(n_users, n_items))


class HybridRec(Recommender):
    """score(item) = sum over models of  weight / (rrf_k + rank + 1),  rank 0 = best unseen item.

    `parts` are ALREADY-FITTED models. Models that have `set_context` (sequence models)
    receive the same context as the Evaluator gives this model.
    """
    name = "Hybrid"

    def __init__(self, parts: dict, weights: dict, data: Dataset, rrf_k: float = 20.0):
        self.parts, self.weights, self.rrf_k = parts, weights, rrf_k
        self.n_users, self.n_items = data.n_users, data.n_items
        self.set_context(data.train)

    def set_context(self, df):
        self.seen = seen_matrix(df, self.n_users, self.n_items)
        for m in self.parts.values():
            if hasattr(m, "set_context"):
                m.set_context(df)

    def fit(self, data):                       # nothing to train: the parts are fitted already
        return self

    def contributions(self, user_ids) -> dict:
        """{part name: (n_users, n_items) points that part gives each item}."""
        users = np.asarray(user_ids)
        seen = self.seen[users].toarray() > 0
        out = {}
        for name, model in self.parts.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            s = np.array(model.score(users), dtype=np.float32, copy=True)
            s[seen] = -np.inf
            rank = np.argsort(np.argsort(-s, axis=1), axis=1)       # owned items rank last
            c = w / (self.rrf_k + rank + 1.0)
            c[seen] = 0.0
            out[name] = c.astype(np.float32)
        return out

    def score(self, user_ids):
        parts = self.contributions(user_ids)
        if not parts:
            return np.zeros((len(np.asarray(user_ids)), self.n_items), dtype=np.float32)
        return np.sum(list(parts.values()), axis=0).astype(np.float32)

    def explain(self, user_id: int, item_id: int, n: int = 4):
        """Share of this item's fused score that each part contributed -> [(part, share)]."""
        c = self.contributions(np.array([user_id]))
        total = sum(float(v[0, item_id]) for v in c.values())
        if total <= 0:
            return []
        return sorted(((k, float(v[0, item_id]) / total) for k, v in c.items()),
                      key=lambda p: -p[1])[:n]
