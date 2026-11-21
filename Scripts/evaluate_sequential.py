"""Run:  python scripts/evaluate_sequential.py     (about 2-4 minutes on CPU)"""
import time

import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.knn import UserKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.sequential import GRU4RecRec, SASRecRec
from recsys.models.wrappers import ColdUserFallback

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)

SPECS = {   # written out explicitly, one config per line
    "GRU4Rec": (GRU4RecRec, [
        dict(dim=32, epochs=20),
        dict(dim=64, epochs=20),
        dict(dim=32, epochs=40),
    ]),
    "SASRec": (SASRecRec, [
        dict(dim=32, layers=1, epochs=20),
        dict(dim=32, layers=2, epochs=20),
        dict(dim=64, layers=1, epochs=20),
        dict(dim=32, layers=1, epochs=40),
    ]),
}


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on the VALIDATION split...\n")
    seq_models = []
    for label, (cls, grid) in SPECS.items():
        _, _, model = tune(cls, grid, data, ev_val, label)
        seq_models.append(model)

    base = [PopularityRec().fit(data), UserKNN(k=400).fit(data), TwoTowerRec(**BEST_TT).fit(data)]
    models = [ColdUserFallback(m, data) for m in base + seq_models]

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    sas = models[-1]
    t0 = time.perf_counter()
    sas.score(ev_test.users[:1000])
    print(f"{sas.name} inference: {(time.perf_counter() - t0) / 1000 * 1000:.2f} ms/user\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_sequential.csv")

    u = int(ev_test.users[0])
    print(f"user {u} recent likes: {sas.seqs[u][-5:].tolist()}")
    for i in sas.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"  -> item {i} ({data.items.loc[i, 'title']}), recent items closest to it: {sas.explain(u, i)}")


if __name__ == "__main__":
    main()
