import numpy as np
import pandas as pd
from scipy import sparse as sp

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.wrappers import ColdUserFallback
from recsys.retrieval.candidates import CandidateGenerator
from recsys.retrieval.index import VectorIndex


def test_index_finds_nearest():
    idx = VectorIndex(np.eye(4, dtype=np.float32))
    _, I = idx.search(np.array([[0, 0, 1, 0]], dtype=np.float32), 2)
    assert I[0][0] == 2


class _Const(Recommender):
    name = "const"

    def fit(self, data):
        return self

    def score(self, user_ids):
        return np.full((len(user_ids), 4), -5.0, dtype=np.float32)


def test_cold_user_gets_popularity_warm_user_untouched():
    train = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 1, 0], "label": [1, 1, 1]})
    empty = train.iloc[:0]
    data = Dataset(train, empty, empty, pd.DataFrame(), n_users=3, n_items=4)
    s = ColdUserFallback(_Const(), data).score(np.array([0, 2]))
    assert np.allclose(s[0], -5.0)                       # warm user keeps the model's scores
    assert np.allclose(s[1], [1.0, 0.5, 0.0, 0.0])       # cold user gets popularity


class _Fake(Recommender):
    name = "fake"

    def __init__(self, n_users, n_items):
        rng = np.random.default_rng(0)
        self.item_codes = rng.normal(size=(n_items, 4)).astype(np.float32)
        self.u = rng.normal(size=(n_users, 4)).astype(np.float32)

    def fit(self, data):
        return self

    def user_codes(self, ids):
        return self.u[np.asarray(ids)]

    def score(self, ids):
        return self.u[np.asarray(ids)] @ self.item_codes.T


def test_candidates_never_include_owned_items():
    n_u, n_i = 3, 30
    fake = _Fake(n_u, n_i)
    seen = np.zeros((n_u, n_i), dtype=np.float32)
    seen[0, :10] = 1
    history = sp.csr_matrix(seen)
    gen = CandidateGenerator(fake, fake, fake, pop=np.arange(n_i),
                             n_tt=5, n_content=5, n_iknn=5, n_pop=5)
    m = gen.generate(np.array([0, 1]), history)
    assert not (m["union"][0] & (seen[0] > 0)).any()
    assert m["tt"][0].sum() == 5
