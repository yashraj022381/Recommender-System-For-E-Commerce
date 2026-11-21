"""Run:  python scripts/evaluate_content.py [--embed]"""
import argparse

import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.models.baselines import PopularityRec
from recsys.models.content import EmbeddingContentRec, TfidfContentRec
from recsys.models.knn import ItemKNN, UserKNN

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)


def cold_item_recall(data, models, k=10):
    """Of test purchases whose ITEM was never seen in training, what share lands in top-k?"""
    test = data.test[(data.test["label"] == 1) & data.test["cold_item"]]
    by_user = test.groupby("user_id")["item_id"].apply(set)
    users = np.array(by_user.index)
    hist = (data.train_csr + data.csr(data.val)).tocsr()
    total = sum(len(v) for v in by_user)
    out = {}
    for m in models:
        found = 0
        s = np.array(m.score(users), dtype=np.float32, copy=True)
        h = hist[users].tocoo()
        s[h.row, h.col] = -np.inf
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        for row, u in enumerate(users):
            found += len(set(top[row]) & by_user[u])
        out[m.name] = found / max(total, 1)
    return pd.Series(out, name=f"cold-item recall@{k}"), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    args = ap.parse_args()

    data = load_dataset()
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    models = [PopularityRec().fit(data),
              UserKNN(k=400).fit(data), ItemKNN(k=400).fit(data),
              TfidfContentRec().fit(data)]
    if args.embed:
        models.append(EmbeddingContentRec().fit(data))

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of items never seen in training ===")
    print(rec.sort_values(ascending=False), "\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_content.csv")

    tf = models[3]
    u = int(ev_test.users[0])
    for i in tf.recommend(u, k=3, exclude=set(ev_test.history[u].indices)):
        print(f"user {u} -> item {i} ({data.items.loc[i, 'title']}): "
              f"matched words {tf.top_terms(u, i)}")


if __name__ == "__main__":
    main()
