"""Run:  python scripts/evaluate_neural.py     (about 2-5 minutes on CPU)"""
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.knn import UserKNN
from recsys.models.neural import NCFRec, TwoTowerRec

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

#SPECS = {
#    "NCF": (NCFRec, [dict(dim=d, epochs=e) for d in (32, 64) for e in (4, 6, 8)]),
#    "TwoTower": (TwoTowerRec, [
#        dict(dim=16, id_dropout=0.3, temperature=t, epochs=e, logq=False)
#                               for t in (0.1, 0.2) for e in (15, 30) for q in (False, True)]),
#}

SPECS = {
    "NCF": (NCFRec, [dict(dim=32, epochs=e) for e in (8, 10, 12)]),
    "TwoTower": (TwoTowerRec, [
        dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=15, logq=False),
        dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=15, logq=True),
        dict(dim=16, id_dropout=0.3, temperature=0.5, epochs=15, logq=False),
        dict(dim=16, id_dropout=0.3, temperature=0.5, epochs=15, logq=True),
        dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8,  logq=True),
        dict(dim=16, id_dropout=0.3, temperature=0.5, epochs=8,  logq=True),
    ]),
}


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on the VALIDATION split...\n")
    models = [PopularityRec().fit(data), UserKNN(k=400).fit(data)]
    for label, (cls, grid) in SPECS.items():
        _, _, model = tune(cls, grid, data, ev_val, label)
        models.append(model)

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_neural.csv")

    tt = models[-1]
    u = int(ev_test.users[0])
    for i in tt.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"user {u} -> item {i} ({data.items.loc[i, 'title']}), "
              f"closest liked items: {tt.explain(u, i)}")


if __name__ == "__main__":
    main()
