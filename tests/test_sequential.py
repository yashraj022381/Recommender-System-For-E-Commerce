import numpy as np
import pandas as pd
import pytest

from recsys.data.loader import Dataset
from recsys.models.sequential import GRU4RecRec, SASRecRec, left_pad, make_windows


def test_windows_cover_every_transition_once():
    targets = []
    for w in make_windows(np.arange(10), 4):
        targets += list(w[1:])
    assert sorted(targets) == list(range(1, 10))


def test_left_pad_puts_newest_last():
    assert left_pad([np.array([3, 4])], 4).tolist() == [[0, 0, 4, 5]]


def cycle_dataset(n_users=30, n_items=6, length=8):
    """Everyone walks the cycle 0->1->2->3->4->5->0..., starting at a different item."""
    rows = [(u, (u + k) % n_items, k, 1) for u in range(n_users) for k in range(length)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "label"])
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, pd.DataFrame(), n_users=n_users, n_items=n_items)


@pytest.mark.parametrize("cls", [GRU4RecRec, SASRecRec])
def test_learns_the_next_item_in_a_cycle(cls):
    m = cls(dim=16, max_len=8, epochs=120, lr=5e-3, batch_size=16, dropout=0.0).fit(cycle_dataset())
    users = np.arange(6)
    assert (m.score(users).argmax(1) == (users + 8) % 6).all()      # last item was (u+7)%6


def test_set_context_changes_the_story():
    m = GRU4RecRec(dim=16, max_len=8, epochs=120, lr=5e-3, batch_size=16, dropout=0.0).fit(cycle_dataset())
    m.set_context(pd.DataFrame({"user_id": [0] * 4, "item_id": [0, 1, 2, 3],
                                "timestamp": [0, 1, 2, 3], "label": [1] * 4}))
    assert m.score(np.array([0]))[0].argmax() == 4                  # story ends at 3 -> next is 4
