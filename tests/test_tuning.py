from recsys.evaluation.tuning import tune


class _Dummy:
    def __init__(self, v):
        self.v = v

    def fit(self, data):
        return self


class _Ev:
    def evaluate(self, model):
        import pandas as pd
        return pd.DataFrame({"ndcg@10": [model.v], "cold_user": [False]})


def test_tune_tries_every_config_and_picks_best():
    cfg, score, model = tune(_Dummy, [dict(v=1), dict(v=5), dict(v=3)], None, _Ev())
    assert cfg == dict(v=5) and score == 5
