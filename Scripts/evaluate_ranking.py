"""Run:  python scripts/evaluate_ranking.py     (about 1-2 minutes)"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.candidates import candidate_report
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.ranker import FeatureBuilder, TwoStageRec
from recsys.models.wrappers import ColdUserFallback
from recsys.retrieval.candidates import CandidateGenerator
from sklearn.inspection import permutation_importance

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    pop_model = PopularityRec().fit(data)
    uknn = UserKNN(k=400).fit(data)
    iknn = ItemKNN(k=400).fit(data)
    content = TfidfContentRec().fit(data)
    tt = TwoTowerRec(**BEST_TT).fit(data)
    pop = np.asarray(data.train_pos_csr.sum(0)).ravel()
    gen = CandidateGenerator(tt, content, iknn, pop)

    # ---------- Stage 1: how good is the sieve? ----------
    users = ev_test.users
    t0 = time.perf_counter()
    masks = gen.generate(users, ev_test.history)
    ms = (time.perf_counter() - t0) / len(users) * 1000
    cold_items = np.asarray(data.train_csr.sum(0)).ravel() == 0
    print(f"=== STAGE 1: candidate recall on TEST (ANN backend: {gen.index.backend}, "
          f"{ms:.2f} ms/user for all sources) ===")
    print(candidate_report(masks, ev_test, users, cold_items), "\n")

    # ---------- Stage 2: train the judge on VALIDATION labels ----------
    has_hist = np.asarray(data.train_pos_csr.sum(1)).ravel() > 0
    tr_users = np.flatnonzero(has_hist)
    fb = FeatureBuilder(data, tt, content, iknn)
    m_val = gen.generate(tr_users, ev_val.history)["union"]
    X, r, c = fb.build(tr_users, m_val, ev_val.history)
    y = np.asarray(ev_val.truth[tr_users[r], c]).ravel()
    print(f"ranker training rows: {len(y):,}   positives: {int(y.sum()):,} ({y.mean():.3%})\n")
    ranker = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                            max_depth=4, random_state=0).fit(X, y)

    cold_i = np.asarray(data.train_pos_csr.sum(0)).ravel() == 0
    is_cold = cold_i[c]
    print(f"val positives on cold items: {int(y[is_cold].sum())} of {int(y.sum())}")
    print(f"candidate rows on cold items: {int(is_cold.sum()):,}   "
          f"positive rate cold={y[is_cold].mean():.3%}  warm={y[~is_cold].mean():.3%}")
    pick = np.random.default_rng(0).choice(len(y), size=min(150_000, len(y)), replace=False)
    pi = permutation_importance(ranker, X[pick], y[pick], scoring="roc_auc",
                                n_repeats=3, random_state=0)
    print("\nfeature importance (drop in AUC when shuffled):")
    for n, v in sorted(zip(FeatureBuilder.NAMES, pi.importances_mean), key=lambda t: -t[1]):
        print(f"  {n:<13} {v:.4f}")
    print()
    
    two_stage = TwoStageRec(gen, fb, ranker, ev_test.history)
    models = [ColdUserFallback(m, data) for m in (pop_model, uknn, tt, two_stage)]
    models[-1].name = "TwoStage(rank)"

    t0 = time.perf_counter()
    models[-1].score(users[:500])
    print(f"two-stage inference: {(time.perf_counter() - t0) / 500 * 1000:.2f} ms/user\n")

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False))

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_ranking.csv")


if __name__ == "__main__":
    main()
