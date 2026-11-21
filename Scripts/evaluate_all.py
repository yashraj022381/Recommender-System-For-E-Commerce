"""Run:  python scripts/evaluate_all.py     (about 2-4 minutes)  -> artifacts/leaderboard.csv
One seed per model; Phase 13A showed seed noise of about +-0.005 NDCG for the neural models."""
import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.evaluation.beyond import beyond_accuracy
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, summary
from recsys.serving.service import RecService

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)


def main():
    svc = RecService()
    data = svc.data
    ev = Evaluator(data, split="test", ks=(10,))
    train_pop = np.asarray(data.train_pos_csr.sum(0)).ravel().astype(np.float64)

    rows = {}
    for name, m in svc.models.items():
        per_user = ev.evaluate(m)
        row = summary(per_user).to_dict()
        row["ndcg@10 (warm)"] = summary(per_user, "warm")["ndcg@10"]
        row["ndcg@10 (cold users)"] = summary(per_user, "cold")["ndcg@10"]
        row.update(beyond_accuracy(m, ev, train_pop, svc.sim_content, svc.cats).to_dict())
        row["latency_ms"] = float(np.median([svc.recommend(name, int(u), 10)[1] for u in ev.users[:30]]))
        rows[name] = row
        print(f"  {name:<34} NDCG@10 = {row['ndcg@10']:.4f}")

    rec, total = cold_item_recall(data, list(svc.models.values()))
    table = pd.DataFrame(rows).T
    table["cold_item_recall@10"] = rec.reindex(table.index)
    table = table.sort_values("ndcg@10", ascending=False)

    acc = ["precision@10", "recall@10", "hit@10", "mrr@10", "ndcg@10", "map@10",
           "ndcg@10 (warm)", "ndcg@10 (cold users)", "cold_item_recall@10"]
    print(f"\n=== ACCURACY (TEST, K=10, {len(ev.users):,} users; cold-item view = {total} purchases) ===")
    print(table[acc], "\n")
    print("=== BEYOND ACCURACY + SPEED ===")
    print(table[["coverage", "novelty", "intra_sim", "distinct_cats", "gini", "latency_ms"]])

    PATHS.make()
    table.to_csv(PATHS.artifacts / "leaderboard.csv")
    print(f"\nSaved {PATHS.artifacts / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
