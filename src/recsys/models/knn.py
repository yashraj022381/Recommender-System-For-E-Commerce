"""Neighborhood collaborative filtering: item-based and user-based."""
import numpy as np
from scipy import sparse as sp

from recsys.data.loader import Dataset
from recsys.models.base import Recommender


def cosine_topk(M: sp.csr_matrix, k: int, shrink: float, block: int = 500) -> sp.csr_matrix:
    """Row-to-row cosine similarity, keeping only the top-k neighbors per row.

    Works in blocks so memory stays small even for large catalogs.
    sim(i, j) = dot(i, j) / (|i| * |j| + shrink)
    """
    M = M.tocsr().astype(np.float32)
    n = M.shape[0]
    norms = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    MT = M.T.tocsr()

    rows, cols, vals = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = (M[s:e] @ MT).toarray()                         # (block, n) dot products
        sim /= (np.outer(norms[s:e], norms) + shrink)          # shrunk cosine
        sim[np.arange(e - s), np.arange(s, e)] = 0.0           # no self-similarity
        kk = min(k, n - 1)
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(sim, top, 1)
        keep = v > 0
        r = np.repeat(np.arange(s, e), kk).reshape(-1, kk)
        rows.append(r[keep]); cols.append(top[keep]); vals.append(v[keep])

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )


class _CF(Recommender):
    def _setup(self, data: Dataset):
        self.R = data.train_pos_csr                       # liked items only
        pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        # tiny popularity tie-breaker: users with no neighbors fall back to popularity
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)


class ItemKNN(_CF):
    """'Toys that are bought by the same kids as the toys you liked.'"""

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink
        self.name = f"ItemKNN(k={k})"

    def fit(self, data: Dataset):
        self._setup(data)
        S = cosine_topk(self.R.T.tocsr(), self.k, self.shrink)   # item x item
        self.S_T = S.T.tocsr()          # score(j) = sum_i liked(i) * sim(j's neighbor i)
        return self

    def score(self, user_ids):
        return (self.R[user_ids] @ self.S_T).toarray() + self.tiebreak

    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Which items the user liked make this item look good? -> [(item_id, sim)]"""
        liked = self.R[user_id].indices
        sims = self.S_T[item_id].toarray().ravel()[:0] if False else None  # (kept simple below)
        row = self.S_T.getrow(item_id)
        d = dict(zip(row.indices, row.data))
        pairs = sorted(((i, d[i]) for i in liked if i in d), key=lambda x: -x[1])
        return pairs[:n]


class UserKNN(_CF):
    """'Kids with a toy box like yours also loved...'"""

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink
        self.name = f"UserKNN(k={k})"

    def fit(self, data: Dataset):
        self._setup(data)
        self.S = cosine_topk(self.R, self.k, self.shrink)        # user x user
        return self

    def score(self, user_ids):
        return (self.S[user_ids] @ self.R).toarray() + self.tiebreak

    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Which similar users liked this item? -> [(user_id, sim)]"""
        row = self.S.getrow(user_id)
        likers = set(self.R[:, item_id].indices)
        pairs = [(u, s) for u, s in zip(row.indices, row.data) if u in likers]
        return sorted(pairs, key=lambda x: -x[1])[:n]
