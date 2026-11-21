import numpy as np
import pandas as pd

from recsys.evaluation.stats import paired_bootstrap


def test_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.random(500) * 0.1)
    a = b + 0.02                                  # a is better for every user
    d, lo, hi, p = paired_bootstrap(a, b)
    assert d > 0 and lo > 0 and p < 0.05


def test_bootstrap_sees_no_difference_in_noise():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.random(500))
    b = pd.Series(rng.random(500))
    _, lo, hi, p = paired_bootstrap(a, b)
    assert lo < 0 < hi
