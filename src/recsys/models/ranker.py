"""Stage 2: features for each (user, candidate) pair + the two-stage recommender."""
import numpy as np

from scipy import sparse as sp
from recsys.models.base import Recommender


class FeatureBuilder:
    NAMES = ["tt", "tt_rank", "content", "content_rank", "iknn", "iknn_rank",
             "item_pop", "user_n", "price", "price_gap", "cat_affinity"]

    def __init__(self, data, tt, content, iknn):
        self.models = (("tt", tt), ("content", content), ("iknn", iknn))
        items = data.items
        R = data.train_pos_csr
        cnt_i = np.asarray(R.sum(0)).ravel()
        cnt_u = np.asarray(R.sum(1)).ravel()
        self.item_pop, self.user_n = np.log1p(cnt_i), np.log1p(cnt_u)
        self.price = np.log1p(items["price"].to_numpy(dtype=float))
        self.user_price = (R @ self.price) / np.maximum(cnt_u, 1)
        self.cat = items["category"].astype("category").cat.codes.to_numpy()
        C = sp.csr_matrix((np.ones(len(self.cat)), (np.arange(len(self.cat)), self.cat)),
                          shape=(len(self.cat), self.cat.max() + 1))
        A = (R @ C).toarray()
        self.aff = A / np.maximum(A.sum(1, keepdims=True), 1)     # share of the user's likes per category
        

    def build(self, users, mask, history):
        users = np.asarray(users)
        seen = history[users].toarray() > 0
        r, c = np.nonzero(mask)
        cols = []
        for _, model in self.models:
            s = np.array(model.score(users), dtype=np.float32, copy=True)
            s[seen] = -np.inf
            rank = np.argsort(np.argsort(-s, axis=1), axis=1)
            cols += [s[r, c], np.log1p(rank[r, c])]
        u = users[r]
        cols += [self.item_pop[c], self.user_n[u], self.price[c],
                np.abs(self.price[c] - self.user_price[u]), self.aff[u, self.cat[c]]]
        return np.column_stack(cols).astype(np.float32), r, c



class TwoStageRec(Recommender):
    """Sieve -> judge. Non-candidates get score -1, so they always rank below candidates."""
    name = "TwoStage"

    def __init__(self, gen, fb, ranker, history):
        self.gen, self.fb, self.ranker, self.history = gen, fb, ranker, history

    def fit(self, data):
        return self

    def score(self, user_ids):
        users = np.asarray(user_ids)
        mask = self.gen.generate(users, self.history)["union"]
        X, r, c = self.fb.build(users, mask, self.history)
        out = np.full(mask.shape, -1.0, dtype=np.float32)
        out[r, c] = self.ranker.predict_proba(X)[:, 1]
        return out
        
        
