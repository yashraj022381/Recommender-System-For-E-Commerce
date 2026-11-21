import numpy as np

from recsys.evaluation.abtest import repeat_ab, simulate


def exp(rel, oos=False, n=200, k=10):
    return {"rel": np.full((n, k), rel), "price": np.full((n, k), 10.0),
            "oos": np.full((n, k), oos)}


def test_loved_toys_earn_more_than_unloved():
    rng = np.random.default_rng(0)
    idx = np.arange(200)
    good = simulate(exp(True), idx, rng, sessions=5)
    bad = simulate(exp(False), idx, rng, sessions=5)
    assert good["revenue"].mean() > 5 * bad["revenue"].mean()


def test_sold_out_toys_cannot_be_bought():
    df = simulate(exp(True, oos=True), np.arange(200), np.random.default_rng(0), sessions=5)
    assert df["revenue"].sum() == 0 and df["lost"].sum() > 0


def test_a_a_test_rarely_cries_wolf():
    any_sig, _ = repeat_ab(exp(True), exp(True), reps=100, sessions=5, seed=1)
    assert any_sig < 0.2                          # nominal 5%; loose bound keeps the test stable
