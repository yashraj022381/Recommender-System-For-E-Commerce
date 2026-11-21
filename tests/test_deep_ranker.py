import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from recsys.models.deep_ranker import DeepRanker


def toy(n=4000, seed=0):
    """'buy' depends on feature 0; 'love' needs a buy AND feature 1 > 0. Last column = category id."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    cat = rng.integers(0, 3, n)
    buy = x[:, 0] + 0.3 * rng.normal(size=n) > 0.5
    like = buy & (x[:, 1] > 0)
    X = np.column_stack([x, cat]).astype(np.float32)
    return X, buy.astype(np.float32), like.astype(np.float32)


@pytest.mark.parametrize("multitask", [True, False])
def test_ranker_learns_the_love_label(multitask):
    X, b, l = toy()
    m = DeepRanker(multitask=multitask, epochs=30, lr=5e-3, hidden=32, batch_size=256).fit(X, b, l)
    assert roc_auc_score(l, m.predict_proba(X)[:, 1]) > 0.8


def test_love_probability_never_exceeds_buy_probability():
    X, b, l = toy()
    m = DeepRanker(multitask=True, epochs=5, hidden=32, batch_size=256).fit(X, b, l)
    t = m.predict_tasks(X)
    assert (t["like"] <= t["buy"] + 1e-6).all()
