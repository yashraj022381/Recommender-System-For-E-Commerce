"""Run:  python scripts/evaluate_gnn.py     (about 2-5 minutes on CPU)"""
import time

import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.gnn import LightGCNRec
from recsys.models.knn import UserKNN
from recsys.models.sequential import SASRecRec
from recsys.models.wrappers import ColdUserFallback

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

SPECS = {   # written out explicitly, one config per line
    "LightGCN": (LightGCNRec, [
        dict(dim=32, layers=1, epochs=40),
        dict(dim=32, layers=2, epochs=40),
        dict(dim=32, layers=3, epochs=40),
        dict(dim=32, layers=2, epochs=80),
        dict(dim=64, layers=2, epochs=40),
    ]),
}


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on the VALIDATION split...\n")
    _, _, gcn = tune(LightGCNRec, SPECS["LightGCN"][1], data, ev_val, "LightGCN")

    sas = SASRecRec(dim=32, layers=1, epochs=40).fit(data)
    base = [PopularityRec().fit(data), UserKNN(k=400).fit(data), sas, gcn]
    models = [ColdUserFallback(m, data) for m in base]

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    t0 = time.perf_counter()
    models[-1].score(ev_test.users[:1000])
    print(f"{gcn.name} inference: {(time.perf_counter() - t0):.2f} ms/user\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_gnn.csv")

    u = int(ev_test.users[0])
    for i in gcn.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"user {u} -> item {i} ({data.items.loc[i, 'title']}), "
              f"closest liked items: {gcn.explain(u, i)}")


if __name__ == "__main__":
    main()
