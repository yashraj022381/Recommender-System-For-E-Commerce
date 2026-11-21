"""Run:  python scripts/evaluate_mf.py     (ALS tuning takes about 1-2 minutes)"""
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec, RandomRec
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.mf import ALSRec, SVDRec

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

SPECS = {
    "UserKNN": (UserKNN, [dict(k=k) for k in (50, 100, 200, 400)]),
    "ItemKNN": (ItemKNN, [dict(k=k) for k in (50, 100, 200, 400)]),
    "SVD": (SVDRec, [dict(factors=f) for f in (4, 16, 32, 64)]),
    "ALS": (ALSRec, [dict(factors=f, alpha=a, reg=r)
                     for f in (8, 16, 32) for a in (10.0, 40.0) for r in (0.1, 1.0)]),
}


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on the VALIDATION split...\n")
    models = [RandomRec().fit(data), PopularityRec().fit(data)]
    for label, (cls, grid) in SPECS.items():
        _, _, model = tune(cls, grid, data, ev_val, label)
        models.append(model)

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_mf.csv")

    als = next(m for m in models if m.name.startswith("ALS"))
    u = int(ev_test.users[0])
    for i in als.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"user {u} -> item {i}, because it looks like items they liked: {als.explain(u, i)}")


if __name__ == "__main__":
    main()
