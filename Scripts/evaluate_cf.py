"""Run:  python scripts/evaluate_cf.py"""
import time

import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, leaderboard, summary
from recsys.models.baselines import PopularityRec, RandomRec
from recsys.models.knn import ItemKNN, UserKNN

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)


def tune(cls, data, ev_val, grid=(20, 50, 100)):
    best_k, best = None, -1
    for k in grid:
        m = cls(k=k).fit(data)
        score = summary(ev_val.evaluate(m))["ndcg@10"]
        print(f"  {cls.__name__} k={k:<4} val NDCG@10 = {score:.4f}")
        if score > best:
            best_k, best = k, score
    return best_k


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on validation split...")
    models = [RandomRec().fit(data), PopularityRec().fit(data)]
    for cls in (UserKNN, ItemKNN):
        k = tune(cls, data, ev_val)
        t0 = time.time()
        models.append(cls(k=k).fit(data))
        print(f"  -> {cls.__name__} best k={k}, fit took {time.time() - t0:.1f}s")

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"\n=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg))

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_cf.csv")

    # peek at one explanation
    item_model = models[-1]
    u = int(ev_test.users[0])
    top = item_model.recommend(u, k=3, exclude=set(ev_test.history[u].indices))
    for i in top:
        print(f"\nuser {u} -> item {i}, because they liked items: {item_model.explain(u, i)}")


if __name__ == "__main__":
    main()
