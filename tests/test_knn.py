import numpy as np
import pytest

from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, summary
from recsys.models.baselines import PopularityRec
from recsys.models.knn import ItemKNN, UserKNN, cosine_topk
from scipy import sparse as sp


def test_cosine_topk_tiny():
    # rows 0 and 1 are identical, row 2 is different
    M = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32))
    S = cosine_topk(M, k=2, shrink=0.0).toarray()
    assert S[0, 1] == pytest.approx(1.0)
    assert S[0, 0] == 0 and S[0, 2] == 0


@pytest.fixture(scope="module")
def setup():
    data = load_dataset()
    return data, Evaluator(data, split="test", ks=(10,))


@pytest.mark.parametrize("cls", [ItemKNN, UserKNN])
def test_knn_beats_popularity(setup, cls):
    data, ev = setup
    pop = summary(ev.evaluate(PopularityRec().fit(data)))["ndcg@10"]
    knn = summary(ev.evaluate(cls(k=50).fit(data)))["ndcg@10"]
    assert knn > pop
