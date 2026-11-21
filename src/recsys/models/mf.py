"""Matrix factorization: truncated SVD and implicit ALS (pure NumPy)."""
import numpy as np
from scipy.sparse.linalg import svds

from recsys.data.loader import Dataset
from recsys.models.base import Recommender


def latent_explain(item_vecs, liked_items, item_id, n=3):
    """Which items the user liked look most like the recommended item in code-space?
    Returns [(item_id, cosine_similarity)]."""
    if len(liked_items) == 0:
        return []
    target = item_vecs[item_id]
    cand = item_vecs[liked_items]
    denom = np.linalg.norm(cand, axis=1) * np.linalg.norm(target) + 1e-9
    sims = cand @ target / denom
    order = np.argsort(-sims)[:n]
    return [(int(liked_items[j]), float(sims[j])) for j in order]


class _MF(Recommender):
    """Shared scoring: score = user_code . item_code (+ tiny popularity tie-breaker)."""

    def _prepare(self, data: Dataset):
        self.R = data.train_pos_csr.astype(np.float32)            # liked items only
        pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)          # cold users fall back to popularity

    def score(self, user_ids):
        return self.user_factors[user_ids] @ self.item_factors.T + self.tiebreak

    def explain(self, user_id: int, item_id: int, n: int = 3):
        return latent_explain(self.item_factors, self.R[user_id].indices, item_id, n)


class SVDRec(_MF):
    """Truncated SVD of the user x item 'liked' matrix. Fast, one shot, no iterations."""

    def __init__(self, factors: int = 32, seed: int = 0):
        self.factors, self.seed = factors, seed
        self.name = f"SVD(f={factors})"

    def fit(self, data: Dataset):
        self._prepare(data)
        k = min(self.factors, min(self.R.shape) - 1)
        v0 = np.random.default_rng(self.seed).random(min(self.R.shape)).astype(np.float32)
        U, s, Vt = svds(self.R, k=k, v0=v0)
        self.user_factors = (U * s).astype(np.float32)             # users x k
        self.item_factors = Vt.T.astype(np.float32)                # items x k
        return self


class ALSRec(_MF):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008).

    A liked item has confidence 1 + alpha; an untouched item is a weak 'probably not' (confidence 1).
    Alternates: solve all user codes with item codes fixed, then all item codes with user codes fixed.
    """

    def __init__(self, factors: int = 32, reg: float = 0.1, alpha: float = 20.0,
                 iters: int = 15, seed: int = 0):
        self.factors, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed
        self.name = f"ALS(f={factors},a={alpha:g})"

    def _solve(self, M, fixed):
        """For every row of sparse M, solve its code given the `fixed` codes of the columns."""
        k = fixed.shape[1]
        YtY = fixed.T @ fixed                                      # shared by all rows
        reg_I = self.reg * np.eye(k)
        out = np.zeros((M.shape[0], k))
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e:
                continue                                           # no history -> zero code (cold)
            Yr = fixed[M.indices[s:e]]                             # codes of the liked items
            extra = self.alpha * M.data[s:e]                       # confidence - 1
            A = YtY + (Yr.T * extra) @ Yr + reg_I
            b = ((1.0 + extra)[:, None] * Yr).sum(0)
            out[r] = np.linalg.solve(A, b)
        return out

    def fit(self, data: Dataset):
        self._prepare(data)
        rng = np.random.default_rng(self.seed)
        R = self.R.astype(np.float64).tocsr()
        Rt = R.T.tocsr()
        X = rng.normal(0, 0.01, (R.shape[0], self.factors))        # user codes
        Y = rng.normal(0, 0.01, (R.shape[1], self.factors))        # item codes
        for _ in range(self.iters):
            X = self._solve(R, Y)
            Y = self._solve(Rt, X)
        self.user_factors = X.astype(np.float32)
        self.item_factors = Y.astype(np.float32)
        return self
