"""Cold-item evaluation view."""
import numpy as np
import pandas as pd


def cold_item_recall(data, models, k=10):
    """Of test purchases whose ITEM was never seen in training, what share lands in top-k?"""
    test = data.test[(data.test["label"] == 1) & data.test["cold_item"]]
    by_user = test.groupby("user_id")["item_id"].apply(set)
    users = np.array(by_user.index)
    hist = (data.train_csr + data.csr(data.val)).tocsr()
    total = sum(len(v) for v in by_user)
    out = {}
    ctx = pd.concat([data.train, data.val], ignore_index=True)
    for m in models:
        if hasattr(m, "set_context"):
            m.set_context(ctx)
        s = np.array(m.score(users), dtype=np.float32, copy=True)
        h = hist[users].tocoo()
        s[h.row, h.col] = -np.inf
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        found = sum(len(set(top[r]) & by_user[u]) for r, u in enumerate(users))
        out[m.name] = found / max(total, 1)
    return pd.Series(out, name=f"cold-item recall@{k}"), total

