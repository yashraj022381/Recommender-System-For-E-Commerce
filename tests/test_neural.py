import numpy as np
import pandas as pd
import pytest

from recsys.data.loader import Dataset
from recsys.models.neural import NCFRec, TwoTowerRec


def tiny_dataset():
    """Two taste blocks. Item 7 is cold (no interactions) but looks like block B by text/category."""
    rows = []
    for u in range(0, 20):
        rows += [(u, i, 1) for i in (0, 1, 2, 3)]
    for u in range(20, 40):
        rows += [(u, i, 1) for i in (4, 5, 6)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    items = pd.DataFrame({
        "item_id": range(8),
        "title": [f"item {i}" for i in range(8)],
        "text": ["dragon quest hero"] * 4 + ["puzzle logic brain"] * 4,
        "category": ["rpg"] * 4 + ["puzzle"] * 4,
        "price": [10.0, 12, 11, 13, 20, 22, 21, 23],
    })
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, items, n_users=40, n_items=8)


def test_ncf_learns_blocks():
    data = tiny_dataset()
    m = NCFRec(dim=8, hidden=16, epochs=30, n_neg=2, batch_size=64).fit(data)
    s_a = m.score(np.array([0]))[0]                 # user 0 likes block A (items 0-3)
    assert s_a[:4].mean() > s_a[4:].mean()


def test_two_tower_scores_cold_item_by_content():
    data = tiny_dataset()
    m = TwoTowerRec(dim=8, epochs=40, batch_size=64, id_dropout=0.5, temperature=0.2).fit(data)
    s = m.score(np.array([25]))[0]                  # user 25 likes puzzle block (items 4-6)
    assert s[7] > s[:4].mean()                      # cold puzzle item 7 beats the RPG items


def test_two_tower_cold_user_gets_popularity():
    data = tiny_dataset()
    data.n_users = 41                                   # user 40 has no history
    m = TwoTowerRec(dim=8, epochs=5, batch_size=64).fit(data)
    s = m.score(np.array([40]))[0]
    assert s[:7].min() > s[7]                           # items with purchases beat the never-bought item 7
