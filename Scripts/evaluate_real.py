"""Run:  python evaluate_real.py [--quick]     (--quick = first config of every grid; start with this)"""
import argparse
import json

import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard, summary
from recsys.evaluation.stats import paired_bootstrap
from recsys.evaluation.tuning import tune
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.generative import GenerativeRec
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.mf import SVDRec
from recsys.models.neural import TwoTowerRec
from recsys.models.rules import FreshFavoriteRule
from recsys.models.sequential import SASRecRec
from recsys.models.wrappers import ColdUserFallback

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)

SPECS = {   # real data is bigger and sparser, so the grids start larger than the synthetic ones
    "UserKNN": (UserKNN, [dict(k=50), dict(k=200), dict(k=400)]),
    "ItemKNN": (ItemKNN, [dict(k=50), dict(k=200), dict(k=400)]),
    "SVD": (SVDRec, [dict(factors=16), dict(factors=32), dict(factors=64)]),
    "TwoTower": (TwoTowerRec, [dict(dim=32, temperature=0.2, epochs=5, logq=True),
                               dict(dim=32, temperature=0.2, epochs=10, logq=True),
                               dict(dim=64, temperature=0.2, epochs=10, logq=True)]),
    "SASRec": (SASRecRec, [dict(dim=64, layers=1, epochs=10, batch_size=64),
                           dict(dim=64, layers=1, epochs=20, batch_size=64)]),
    "Generative": (GenerativeRec, [dict(dim=64, layers=1, epochs=10, batch_size=64),
                                   dict(dim=64, layers=1, epochs=20, batch_size=64)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    data = load_dataset()
    if json.loads((PATHS.processed / "meta.json").read_text()).get("synthetic"):
        raise SystemExit("The processed data is synthetic. Run prepare_data.py WITHOUT --synthetic first.")
    wrap = lambda m: ColdUserFallback(m, data)
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev = Evaluator(data, split="test", ks=(5, 10, 20))
    print(f"graded test users: {len(ev.users):,} (cold: {int(ev.cold_user[ev.users].sum()):,})\n")

    print("Tuning on the VALIDATION split...\n")
    models = [wrap(PopularityRec().fit(data))]
    for label, (cls, grid) in SPECS.items():
        _, _, m = tune(cls, grid[:1] if args.quick else grid, data, ev_val, label)
        models.append(wrap(m))
    models.append(wrap(FreshFavoriteRule(data)))

    results = {m.name: ev.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False), "\n")

    best = max(results, key=lambda n: summary(results[n])["ndcg@10"])
    print(f"=== PAIRED BOOTSTRAP on NDCG@10 (best model: {best}) ===")
    for name, df in results.items():
        row = {}
        for ref in ("Popularity", best):
            if name != ref:
                d, lo, hi, p = paired_bootstrap(df["ndcg@10"], results[ref]["ndcg@10"])
                row[f"vs {ref}"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"  {name:<26} {row}")

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_real.csv")
    print(f"\nSaved {PATHS.artifacts / 'results_real.csv'}")


if __name__ == "__main__":
    main()
