"""Ranking metrics. Inputs are per-user, computed for a whole batch at once.

hits  : (n_users, K) bool/0-1 matrix. hits[u, j] = 1 if the item at rank j is relevant.
n_rel : (n_users,) number of relevant (held-out) items per user. Must be >= 1.
"""

import numpy as np

METRIC_NAMES = ["precision", "recall", "hit", "mrr", "ndcg", "map"]

def compute_metrics(hits: np.ndarray, n_rel: np.ndarray, k: int) -> dict[str, np.ndarray]:
    hits = hits[:, :k].astype(np.float32)
    n_rel = n_rel.astype(np.float32)
    n_hit = hits.sum(1)


    # discount for rank r (1-based) is 1 / log2(r + 1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    ideal_len = np.minimum(n_rel, k).astype(int)        # best case: all relevant first
    idcg = np.cumsum(disc)[ideal_len - 1]


    ranks = np.arange(1, k + 1)
    prec_at_rank = np.cumsum(hits, axis=1) / ranks
    ap = (prec_at_rank * hits).sum(1) / np.minimum(n_rel, k)
    

    first_hit = hits.argmax(1)
    mrr = np.where(n_hit > 0, 1.0 / (first_hit + 1), 0.0)


    return {
        "precision": n_hit / k,
        "recall": n_hit / n_rel,
        "hit": (n_hit > 0).astype(np.float32),
        "mrr": mrr.astype(np.float32),
        "ndcg": dcg / idcg,
        "map": ap,
    }
