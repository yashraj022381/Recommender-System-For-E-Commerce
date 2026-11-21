"""Business-logic layer: stock filter, diversity (MMR), category cap, reserved slots for new items."""
import numpy as np

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.hybrid import seen_matrix


def simulate_stock(n_items: int, rate: float = 0.05, seed: int = 7) -> np.ndarray:
    """SIMULATED out-of-stock flags. The dataset has no inventory data."""
    return np.random.default_rng(seed).random(n_items) < rate


class BusinessReranker(Recommender):
    """Wraps a base model. Takes its top `pool` unseen, in-stock items and rebuilds the top `depth`.

    lam            1.0 = pure relevance; lower = more variety (MMR: lam*relevance - (1-lam)*similarity)
    cat_cap        max items per category among the first 10 (0 = off)
    fresh_slots    guarantee at least this many NEW items (no training history) in the top 10
    fresh_positions 0-based positions where missing new items are inserted
    hide_oos       drop out-of-stock items

    The output score matrix puts the final list first, in order; every other item scores -1e6.
    item_sim is a dense item x item matrix (fine for ~1k items; at scale compute pool x pool on the fly).
    """

    def __init__(self, base, data: Dataset, item_sim, cats, fresh, oos=None, lam=1.0, cat_cap=0,
                 fresh_slots=0, fresh_positions=(2, 6), hide_oos=False, pool=50, depth=20,
                 fresh_pool=10, name=None):
        self.base = base
        self.n_users, self.n_items = data.n_users, data.n_items
        self.sim = np.asarray(item_sim, dtype=np.float32)
        self.cats = np.asarray(cats, dtype=np.int64)
        self.n_cats = int(self.cats.max()) + 1
        self.fresh = np.asarray(fresh, dtype=bool)
        self.oos = np.zeros(self.n_items, dtype=bool) if oos is None else np.asarray(oos, dtype=bool)
        self.lam, self.cat_cap, self.fresh_slots = lam, cat_cap, fresh_slots
        self.fresh_positions, self.hide_oos = tuple(fresh_positions), hide_oos
        self.pool, self.depth, self.fresh_pool = pool, depth, fresh_pool
        self.name = name or (f"Rerank(lam={lam:g},cap={cat_cap},fresh={fresh_slots}"
                             f"{',stock' if hide_oos else ''})")
        self.set_context(data.train)

    def set_context(self, df):
        self.seen = seen_matrix(df, self.n_users, self.n_items)
        if hasattr(self.base, "set_context"):
            self.base.set_context(df)

    def fit(self, data):
        return self

    # ---- one user ----
    def _greedy(self, pool, rel, S, pcat):
        n = len(pool)
        taken = np.zeros(n, dtype=bool)
        max_sim = np.zeros(n, dtype=np.float32)
        cat_n = np.zeros(self.n_cats, dtype=np.int64)
        out = []
        for _ in range(min(self.depth, n)):
            obj = self.lam * rel - (1.0 - self.lam) * max_sim
            obj[taken] = -np.inf
            if self.cat_cap and len(out) < 10:
                capped = cat_n[pcat] >= self.cat_cap
                if (~capped & ~taken).any():                # if every option is capped, relax the cap
                    obj[capped] = -np.inf
            j = int(obj.argmax())
            taken[j] = True
            out.append(int(pool[j]))
            max_sim = np.maximum(max_sim, S[j])
            cat_n[pcat[j]] += 1
        return out

    def _add_fresh(self, chosen, s):
        need = self.fresh_slots - int(self.fresh[chosen[:10]].sum())
        if need <= 0:
            return chosen
        cand = np.flatnonzero(self.fresh & np.isfinite(s))
        if len(cand) == 0:
            return chosen
        k = min(self.fresh_pool, len(cand))
        top = cand[np.argpartition(-s[cand], k - 1)[:k]]
        top = top[np.argsort(-s[top])]
        have = set(chosen)
        extra = [int(i) for i in top if int(i) not in have][:need]
        for pos, item in zip(self.fresh_positions, extra):
            chosen.insert(min(pos, len(chosen)), item)
        return chosen

    def _rerank_one(self, s):
        p = min(self.pool, int(np.isfinite(s).sum()))
        pool = np.argpartition(-s, p - 1)[:p]
        pool = pool[np.argsort(-s[pool])]
        rel = s[pool]
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        chosen = self._greedy(pool, rel, self.sim[np.ix_(pool, pool)], self.cats[pool])
        if self.fresh_slots:
            chosen = self._add_fresh(chosen, s)
        return chosen[:self.depth]

    # ---- batch ----
    def score(self, user_ids):
        users = np.asarray(user_ids)
        s = np.array(self.base.score(users), dtype=np.float32, copy=True)
        blocked = self.seen[users].toarray() > 0
        if self.hide_oos:
            blocked |= self.oos[None, :]
        s[blocked] = -np.inf
        out = np.full(s.shape, -1e6, dtype=np.float32)
        for r in range(len(users)):
            chosen = self._rerank_one(s[r])
            out[r, chosen] = len(chosen) - np.arange(len(chosen))
        return out
