import numpy as np
import pandas as pd

from recsys.data.loader import Dataset
from recsys.models.gnn import LightGCNRec, normalized_adjacency


def tiny_dataset():
    """Users 0-9 like items 0-3, users 10-19 like items 4-7. User 0 is missing item 3."""
    rows = []
    for u in range(10):
        rows += [(u, i, 1) for i in ((0, 1, 2) if u == 0 else (0, 1, 2, 3))]
    for u in range(10, 20):
        rows += [(u, i, 1) for i in (4, 5, 6, 7)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, pd.DataFrame(), n_users=20, n_items=8)


def test_adjacency_is_symmetric():
    data = tiny_dataset()
    A = normalized_adjacency(data.train_pos_csr).to_dense().numpy()
    assert A.shape == (28, 28)
    assert np.allclose(A, A.T)


def test_lightgcn_finds_the_hidden_taste_block():
    data = tiny_dataset()
    m = LightGCNRec(dim=8, layers=2, epochs=100, batch_size=64, lr=1e-2).fit(data)
    s = m.score(np.array([0]))[0]
    assert s[3] > s[4:].max()
