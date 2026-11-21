"""A one-line baseline: show new items from the user's favorite categories."""
import numpy as np

from recsys.data.loader import Dataset
from recsys.models.base import Recommender


class FreshFavoriteRule(Recommender):
    """Items with no training history, ordered by (share of the user's likes in the item's category).
    Everything else is ordered by popularity. No learning at all."""
    name = "Rule: fresh x favorite"

    def __init__(self, data: Dataset):
        self.cat = data.items["category"].astype("category").cat.codes.to_numpy()
        R = data.train_pos_csr
        pop = np.asarray(R.sum(0)).ravel().astype(np.float32)
        self.pop = pop / max(pop.max(), 1.0)
        self.fresh = pop == 0
        n_c = int(self.cat.max()) + 1
        A = np.zeros((data.n_users, n_c), dtype=np.float32)
        coo = R.tocoo()
        np.add.at(A, (coo.row, self.cat[coo.col]), 1.0)
        self.aff = A / np.maximum(A.sum(1, keepdims=True), 1.0)

    def fit(self, data):
        return self

    def score(self, user_ids):
        u = np.asarray(user_ids)
        a = self.aff[u][:, self.cat]                          # (users, items) category affinity
        s = self.pop[None, :] * 0.5 + a * 0.5                 # warm items: blend of popularity and affinity
        s = np.where(self.fresh[None, :], 2.0 + a, s)         # new items come first, favorite aisle first
        return s.astype(np.float32)
