"""Run:  python scripts/evaluate_generative.py     (about 5-9 minutes on CPU)"""
import time

import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard, summary
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.generative import GenerativeRec
from recsys.models.knn import UserKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.sequential import SASRecRec
from recsys.models.wrappers import ColdUserFallback

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)

GRID = [   # written out explicitly, one config per line
    dict(dim=32, layers=1, epochs=40),
    dict(dim=64, layers=1, epochs=40),
    dict(dim=32, layers=2, epochs=40),
    dict(dim=32, layers=1, epochs=40, codebook=8),
    dict(dim=32, layers=1, epochs=80),
]


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Tuning on the VALIDATION split...\n")
    cfg, _, gen = tune(GenerativeRec, GRID, data, ev_val, "Generative")
    print(f"semantic-ID vocabulary per level: {gen.sizes}   "
          f"unique IDs: {len({tuple(r) for r in gen.codes})} of {data.n_items}\n")

    sas = SASRecRec(dim=32, layers=1, epochs=40).fit(data)
    base = [PopularityRec().fit(data), UserKNN(k=400).fit(data),
            TwoTowerRec(**BEST_TT).fit(data), sas, gen]
    models = [ColdUserFallback(m, data) for m in base]

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    t0 = time.perf_counter()
    models[-1].score(ev_test.users[:500])
    print(f"{gen.name} inference: {(time.perf_counter() - t0) / 500 * 1000:.2f} ms/user\n")

    u = int(ev_test.users[0])
    print(f"user {u} recent likes: {gen.seqs[u][-5:].tolist()}")
    for i in gen.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"  -> item {i} ({data.items.loc[i, 'title']}), address {gen.semantic_id(i)}, "
              f"recent items sharing its address prefix: {gen.explain(u, i)}")
    print()

    # ---- seed check: is the SASRec vs Generative gap bigger than run-to-run noise? ----
    print("=== SEED CHECK (test NDCG@10, 3 seeds each) ===")
    rows = {}
    for label, make, m0 in (
        ("SASRec", lambda s: SASRecRec(dim=32, layers=1, epochs=40, seed=s), sas),
        ("Generative", lambda s: GenerativeRec(**cfg, seed=s), gen),
    ):
        vals = [summary(results[m0.name])["ndcg@10"]]
        for s in (1, 2):
            m = ColdUserFallback(make(s).fit(data), data)
            vals.append(summary(ev_test.evaluate(m))["ndcg@10"])
        rows[label] = {"seed0": vals[0], "seed1": vals[1], "seed2": vals[2],
                       "mean": np.mean(vals), "std": np.std(vals)}
    print(pd.DataFrame(rows).T)

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_generative.csv")


if __name__ == "__main__":
    main()
