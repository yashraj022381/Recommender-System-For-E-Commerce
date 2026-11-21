"""Run:  python scripts/evaluate_baselines.py"""
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.models.baselines import PopularityRec, RandomRec

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)


def main():
    data = load_dataset()
    ev = Evaluator(data, split="test", ks=(5, 10, 20))
    print(f"graded users: {len(ev.users):,}  "
          f"(cold users among them: {ev.cold_user[ev.users].sum():,})\n")

    results = {}
    for model in (RandomRec(), PopularityRec()):
        model.fit(data)
        results[model.name] = ev.evaluate(model)

    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_baselines.csv")


if __name__ == "__main__":
    main()
