"""Fast batch evaluator + leaderboard."""
import numpy as np
import pandas as pd

from recsys.data.loader import Dataset
from recsys.evaluation.metrics import METRIC_NAMES, compute_metrics
from recsys.models.base import Recommender


class Evaluator:
    """Grades any Recommender on the val or test split.

    - Relevant items = held-out interactions with a positive label (rating >= 4).
    - Items the user already has (history) are removed from the ranking.
    - Only users with at least one relevant held-out item are graded.
    """

    def __init__(self, data: Dataset, split: str = "test",
                 ks=(5, 10, 20), batch_size: int = 1024):
        assert split in ("val", "test")
        self.ks = tuple(sorted(ks))
        self.kmax = self.ks[-1]
        self.batch_size = batch_size
        self.n_items = data.n_items

        target = data.val if split == "val" else data.test
        history = data.train_csr
        if split == "test":                       # user also owns what they got in val
            history = (history + data.csr(data.val)).tocsr()
        self.history = history
        self.context = data.train if split == "val" else pd.concat([data.train, data.val], ignore_index=True)
        self.truth = data.csr(target, positives_only=True)

        self.n_rel = np.asarray(self.truth.sum(1)).ravel().astype(int)
        self.users = np.flatnonzero(self.n_rel > 0)
        self.cold_user = data.train_csr.getnnz(axis=1) == 0

    # ---- helpers ----
    def _topk(self, scores: np.ndarray) -> np.ndarray:
        k = self.kmax
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
        return np.take_along_axis(part, order, 1)

    def _hits(self, users: np.ndarray, top: np.ndarray) -> np.ndarray:
        rel = self.truth[users]
        rows = np.repeat(np.arange(len(users)), self.kmax)
        return np.asarray(rel[rows, top.ravel()]).reshape(len(users), self.kmax)

    # ---- main ----
    def evaluate(self, model: Recommender) -> pd.DataFrame:
        """Returns a per-user DataFrame (index = user_id) of metric@k columns."""
        if hasattr(model, "set_context"):          # sequence models read the history up to this split
            model.set_context(self.context)
            
        chunks = []
        for s in range(0, len(self.users), self.batch_size):
            u = self.users[s:s + self.batch_size]
            scores = np.array(model.score(u), dtype=np.float32, copy=True)
            h = self.history[u].tocoo()
            scores[h.row, h.col] = -np.inf          # hide already-owned items
            top = self._topk(scores)
            hits = self._hits(u, top)
            cols = {}
            for k in self.ks:
                for name, v in compute_metrics(hits, self.n_rel[u], k).items():
                    cols[f"{name}@{k}"] = v
            chunks.append(pd.DataFrame(cols, index=u))
        per_user = pd.concat(chunks)
        per_user["cold_user"] = self.cold_user[per_user.index]
        return per_user


def summary(per_user: pd.DataFrame, segment: str = "all") -> pd.Series:
    """segment: 'all' | 'warm' (seen in train) | 'cold' (never seen in train)."""
    if segment == "warm":
        per_user = per_user[~per_user["cold_user"]]
    elif segment == "cold":
        per_user = per_user[per_user["cold_user"]]
    return per_user.drop(columns="cold_user").mean()


def leaderboard(results: dict[str, pd.DataFrame], k: int = 10,
                segment: str = "all") -> pd.DataFrame:
    cols = [f"{m}@{k}" for m in METRIC_NAMES]
    table = pd.DataFrame({name: summary(df, segment)[cols] for name, df in results.items()}).T
    return table.sort_values(f"ndcg@{k}", ascending=False)
