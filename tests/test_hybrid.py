import numpy as np
import pandas as pd

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.hybrid import HybridRec
from recsys.serving.business import BusinessReranker


class _Fixed(Recommender):
    name = "fixed"

    def __init__(self, S):
        self.S = np.asarray(S, dtype=np.float32)

    def fit(self, data):
        return self

    def score(self, user_ids):
        return self.S[np.asarray(user_ids)]


def make_data(n_items, owned=()):
    rows = [(0, i, 1) for i in owned]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "label"]).astype("int64")
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, pd.DataFrame(), n_users=1, n_items=n_items)


A = [[5, 4, 3, 2, 1]]          # likes items in order 0,1,2,3,4
B = [[1, 2, 5, 3, 4]]          # likes item 2 most


def test_fusion_rewards_consensus():
    data = make_data(5)
    parts = {"a": _Fixed(A), "b": _Fixed(B)}
    solo = HybridRec(parts, {"a": 1, "b": 0}, data, rrf_k=0).score([0])[0]
    both = HybridRec(parts, {"a": 1, "b": 1}, data, rrf_k=0).score([0])[0]
    assert solo.argmax() == 0
    assert both.argmax() == 2


def test_fusion_gives_owned_items_zero():
    data = make_data(5, owned=(2,))
    s = HybridRec({"a": _Fixed(A)}, {"a": 1}, data, rrf_k=0).score([0])[0]
    assert s[2] == 0 and (np.delete(s, 2) > 0).all()


def rr(**kw):
    base = _Fixed(np.arange(10, 0, -1)[None, :])               # item 0 best ... item 9 worst
    args = dict(item_sim=np.eye(10, dtype=np.float32), cats=np.zeros(10, dtype=np.int64),
                fresh=np.zeros(10, dtype=bool), pool=10, depth=5)
    args.update(kw)
    return BusinessReranker(base, make_data(10), **args)


def top(m):
    return np.argsort(-m.score([0])[0])[:5].tolist()


def test_out_of_stock_items_are_hidden():
    oos = np.zeros(10, dtype=bool)
    oos[0] = True
    assert top(rr(oos=oos, hide_oos=True)) == [1, 2, 3, 4, 5]


def test_category_cap():
    cats = np.array([0] * 4 + [1] * 3 + [2] * 3)       # 3 aisles, so a cap of 2 can be honored for 5 slots
    assert top(rr(cats=cats, cat_cap=2)) == [0, 1, 4, 5, 7]


def test_category_cap_relaxes_when_every_aisle_is_full(): 
    cats = np.array([0] * 5 + [1] * 5)                 # only 2 aisles: 5 slots can't fit under a cap of 2
    assert top(rr(cats=cats, cat_cap=2)) == [0, 1, 5, 6, 2]

def test_fresh_slot_is_inserted():
    fresh = np.zeros(10, dtype=bool)
    fresh[9] = True
    assert top(rr(fresh=fresh, fresh_slots=1, fresh_positions=(1,))) == [0, 9, 1, 2, 3]
