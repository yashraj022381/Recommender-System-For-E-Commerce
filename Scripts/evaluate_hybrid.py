"""Run:  python scripts/evaluate_hybrid.py     (about 3-6 minutes on CPU)"""
import time

import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.evaluation.list_stats import list_stats
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.generative import GenerativeRec
from recsys.models.hybrid import HybridRec
from recsys.models.knn import UserKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.wrappers import ColdUserFallback
from recsys.serving.business import BusinessReranker, simulate_stock

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)

WEIGHTS = [
    dict(gen=1.0, uknn=0.0,  tt=0.0,  content=0.0),
    dict(gen=1.0, uknn=0.25, tt=0.0,  content=0.0),
    dict(gen=1.0, uknn=0.0,  tt=0.25, content=0.0),
    dict(gen=1.0, uknn=0.25, tt=0.25, content=0.1),
    dict(gen=4.0, uknn=1.0,  tt=1.0,  content=0.5),
]

def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    print("Fitting the four helpers (about 1-2 minutes)...")
    parts = {
        "gen": GenerativeRec(dim=64, layers=1, epochs=40).fit(data),
        "uknn": UserKNN(k=400).fit(data),
        "tt": TwoTowerRec(**BEST_TT).fit(data),
        "content": TfidfContentRec().fit(data),
    }

    print("\nTuning blend weights on the VALIDATION split...\n")
    make = lambda **w: ColdUserFallback(HybridRec(parts, w, data), data)
    best_w, _, _ = tune(make, WEIGHTS, data, ev_val, "Hybrid")

    # ---- shop resources (stock is SIMULATED) ----
    X = parts["content"].X
    item_sim = (X @ X.T).toarray().astype(np.float32)
    cats = data.items["category"].astype("category").cat.codes.to_numpy()
    fresh = np.asarray(data.train_csr.sum(0)).ravel() == 0        # new items: no training history
    oos = simulate_stock(data.n_items)
    print(f"new items: {int(fresh.sum())}   simulated out-of-stock items: {int(oos.sum())}\n")

    hybrid = HybridRec(parts, best_w, data)
    hyb = ColdUserFallback(hybrid, data)

    def rerank(name, **kw):
        return BusinessReranker(hyb, data, item_sim, cats, fresh, oos, name=name, **kw)

    models = [
        ColdUserFallback(PopularityRec().fit(data), data),
        ColdUserFallback(parts["gen"], data),
        ColdUserFallback(parts["uknn"], data),
        hyb,
        rerank("Hybrid+MMR", lam=0.7),
        rerank("Hybrid+Cap", cat_cap=5),
        rerank("Hybrid+Fresh", fresh_slots=2),
        rerank("Hybrid+All", lam=0.7, cat_cap=5, fresh_slots=2),
        rerank("Hybrid+Stock*", hide_oos=True),
    ]

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")
    print("* simulated random stock flags also hide items that WERE bought, so its accuracy drop is partly an artifact.\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    print("=== WHAT THE SHELF LOOKS LIKE (top-10 lists, TEST) ===")
    print(pd.DataFrame({m.name: list_stats(m, ev_test, item_sim, cats, fresh, oos=oos)
                        for m in models[3:]}).T, "\n")

    t0 = time.perf_counter()
    models[7].score(ev_test.users[:500])
    print(f"{models[7].name} inference: {(time.perf_counter() - t0) / 500 * 1000:.2f} ms/user\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_hybrid.csv")

    final = models[7]
    final.set_context(ev_test.context)
    u = int(ev_test.users[0])
    print(f"user {u}: Hybrid+All top-3")
    for i in final.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        tag = " [NEW]" if fresh[i] else ""
        print(f"  -> item {i} ({data.items.loc[i, 'title']}){tag}, blend shares: {hybrid.explain(u, i)}")


if __name__ == "__main__":
    main()
