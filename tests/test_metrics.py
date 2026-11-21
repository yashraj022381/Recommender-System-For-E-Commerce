import numpy as np
import pytest

from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, summary
from recsys.evaluation.metrics import compute_metrics
from recsys.models.base import Recommender
from recsys.models.baselines import PopularityRec, RandomRec


def test_metrics_by_hand():
    # robot shows [a, b, c]; the kid wanted only b  -> hits = [0, 1, 0]
    m = compute_metrics(np.array([[0, 1, 0]]), np.array([1]), k=3)
    assert m["precision"][0] == pytest.approx(1 / 3)
    assert m["recall"][0] == pytest.approx(1.0)
    assert m["hit"][0] == 1
    assert m["mrr"][0] == pytest.approx(0.5)
    assert m["ndcg"][0] == pytest.approx(1 / np.log2(3))
    assert m["map"][0] == pytest.approx(0.5)


def test_no_hits_is_zero():
    m = compute_metrics(np.zeros((1, 5)), np.array([2]), k=5)
    assert all(v[0] == 0 for v in m.values())


class Oracle(Recommender):
    """Cheats by looking at the answers -> must score (almost) perfectly."""
    name = "oracle"

    def __init__(self, truth):
        self.truth = truth

    def fit(self, data):
        return self

    def score(self, user_ids):
        return self.truth[user_ids].toarray()


@pytest.fixture(scope="module")
def setup():
    data = load_dataset()
    return data, Evaluator(data, split="test", ks=(10,))


def test_oracle_is_perfect(setup):
    data, ev = setup
    res = summary(ev.evaluate(Oracle(ev.truth)))
    assert res["ndcg@10"] == pytest.approx(1.0, abs=1e-5)
    assert res["hit@10"] == pytest.approx(1.0)


def test_popularity_beats_random(setup):
    data, ev = setup
    r = summary(ev.evaluate(RandomRec().fit(data)))["ndcg@10"]
    p = summary(ev.evaluate(PopularityRec().fit(data)))["ndcg@10"]
    assert p > r
