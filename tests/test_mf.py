import numpy as np
import pandas as pd
import pytest

from recsys.data.loader import Dataset, load_dataset
from recsys.evaluation.evaluator import Evaluator, summary
from recsys.models.baselines import PopularityRec
from recsys.models.mf import ALSRec, SVDRec


def tiny_dataset():
    """Two taste blocks: users 0-2 like items 0-3, users 3-5 like items 4-7.
    User 0 is missing item 3, which a good model should still rank above items 4-7."""
    rows = []
    for u, items in {0: [0, 1, 2], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3],
                     3: [4, 5, 6, 7], 4: [4, 5, 6, 7], 5: [4, 5, 6, 7]}.items():
        rows += [(u, i, 1) for i in items]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, pd.DataFrame(), n_users=6, n_items=8)


@pytest.mark.parametrize("model", [SVDRec(factors=2), ALSRec(factors=4, iters=10)])
def test_mf_finds_the_hidden_taste_block(model):
    data = tiny_dataset()
    s = model.fit(data).score(np.array([0]))[0]
    assert s[3] > s[4:].max()


@pytest.fixture(scope="module")
def setup():
    data = load_dataset()
    return data, Evaluator(data, split="test", ks=(10,))


@pytest.mark.parametrize("model", [SVDRec(factors=32), ALSRec(factors=32, alpha=20.0)])
def test_mf_beats_popularity(setup, model):
    data, ev = setup
    pop = summary(ev.evaluate(PopularityRec().fit(data)))["ndcg@10"]
    mf = summary(ev.evaluate(model.fit(data)))["ndcg@10"]
    assert mf > pop
