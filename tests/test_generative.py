import numpy as np
import pandas as pd

from recsys.data.loader import Dataset
from recsys.models.generative import GenerativeRec, semantic_ids

FRUITS = ["apple", "banana", "cherry", "grape", "lemon", "mango"]


def test_semantic_ids_are_unique():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 16))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = semantic_ids(v, levels=2, codebook=4)          # few clusters -> collisions -> tie-break token
    assert ids.shape == (200, 3)
    assert len({tuple(r) for r in ids}) == 200


def cycle_dataset(n_users=30, n_items=6, length=8):
    """Everyone walks the cycle 0->1->...->5->0, starting at a different item."""
    rows = [(u, (u + k) % n_items, k, 1) for u in range(n_users) for k in range(length)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "label"])
    items = pd.DataFrame({"item_id": range(n_items), "title": FRUITS,
                          "text": [f"{f} fruit" for f in FRUITS], "category": "fruit",
                          "price": [5.0, 6, 7, 8, 9, 10]})
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, items, n_users=n_users, n_items=n_items)


def test_learns_the_next_item_in_a_cycle():
    m = GenerativeRec(dim=16, layers=1, max_len=8, levels=2, codebook=6, epochs=200,
                      lr=5e-3, batch_size=16, dropout=0.0).fit(cycle_dataset())
    users = np.arange(6)
    assert (m.score(users).argmax(1) == (users + 8) % 6).all()      # last item was (u+7)%6


def test_scores_are_log_probabilities_of_valid_addresses():
    m = GenerativeRec(dim=8, layers=1, max_len=8, levels=2, codebook=6, epochs=3,
                      batch_size=16).fit(cycle_dataset())
    s = m.score(np.array([0, 1]))
    assert (s <= 0).all()
    assert np.exp(s).sum(1).max() <= 1 + 1e-4       # total probability of valid addresses is at most 1
