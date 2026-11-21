"""Run:  python scripts/evaluate_deep_ranker.py     (about 3-6 minutes on CPU)"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, leaderboard
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.deep_ranker import DeepRanker, FeatureBuilderCat
from recsys.models.knn import ItemKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.ranker import TwoStageRec
from recsys.models.wrappers import ColdUserFallback
from recsys.retrieval.candidates import CandidateGenerator

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)


class _Trim:
    """The GBDT never sees the category-id column (same inputs as Phase 7)."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, :-1])


def pick_epochs(multitask, X, y_buy, y_like, row_users, max_epochs=12):
    """Hold out 15% of USERS; return the epoch count with the best held-out 'love' AUC."""
    rng = np.random.default_rng(0)
    uniq = np.unique(row_users)
    hold = rng.choice(uniq, size=int(0.15 * len(uniq)), replace=False)
    is_hold = np.isin(row_users, hold)
    m = DeepRanker(multitask=multitask, epochs=max_epochs).fit(
        X[~is_hold], y_buy[~is_hold], y_like[~is_hold], monitor=(X[is_hold], y_like[is_hold]))
    best = int(np.argmax(m.history)) + 1
    curve = " ".join(f"{a:.3f}" for a in m.history)
    print(f"  {'multi ' if multitask else 'single'} held-out AUC by epoch: {curve}  -> use {best} epochs")
    return best


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev_test = Evaluator(data, split="test", ks=(5, 10, 20))

    pop_model = PopularityRec().fit(data)
    iknn = ItemKNN(k=400).fit(data)
    content = TfidfContentRec().fit(data)
    tt = TwoTowerRec(**BEST_TT).fit(data)
    pop = np.asarray(data.train_pos_csr.sum(0)).ravel()
    gen = CandidateGenerator(tt, content, iknn, pop)
    fb = FeatureBuilderCat(data, tt, content, iknn)

    # ---------- training rows come from the VALIDATION window ----------
    has_hist = np.asarray(data.train_pos_csr.sum(1)).ravel() > 0
    tr_users = np.flatnonzero(has_hist)
    m_val = gen.generate(tr_users, ev_val.history)["union"]
    X, r, c = fb.build(tr_users, m_val, ev_val.history)
    row_users = tr_users[r]
    y_like = np.asarray(ev_val.truth[row_users, c]).ravel()
    y_buy = np.asarray(data.csr(data.val)[row_users, c]).ravel()
    print(f"ranker rows: {len(y_like):,}   buy positives: {int(y_buy.sum()):,} ({y_buy.mean():.3%})   "
          f"love positives: {int(y_like.sum()):,} ({y_like.mean():.3%})   "
          f"love|buy: {y_like.sum() / max(y_buy.sum(), 1):.1%}\n")

    print("Choosing epochs on held-out users...")
    ep_single = pick_epochs(False, X, y_buy, y_like, row_users)
    ep_multi = pick_epochs(True, X, y_buy, y_like, row_users)
    print()

    gbdt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                          random_state=0).fit(X[:, :-1], y_like)
    single = DeepRanker(multitask=False, epochs=ep_single).fit(X, y_buy, y_like)
    multi = DeepRanker(multitask=True, epochs=ep_multi).fit(X, y_buy, y_like)
    rankers = {"TwoStage-GBDT": _Trim(gbdt), "TwoStage-DeepSingle": single,
               "TwoStage-DeepMulti": multi}

    # ---------- ranking quality on candidate rows (TEST) ----------
    users = ev_test.users
    m_te = gen.generate(users, ev_test.history)["union"]
    Xt, rt, ct = fb.build(users, m_te, ev_test.history)
    yl_t = np.asarray(ev_test.truth[users[rt], ct]).ravel()
    yb_t = np.asarray(data.csr(data.test)[users[rt], ct]).ravel()
    rows = {}
    for name, rk in rankers.items():
        p = rk.predict_proba(Xt)[:, 1]
        rows[name] = {"AUC love": roc_auc_score(yl_t, p), "AUC buy (using love score)": roc_auc_score(yb_t, p)}
    rows["DeepMulti's own buy head"] = {"AUC love": np.nan,
                                        "AUC buy (using love score)": roc_auc_score(yb_t, multi.predict_tasks(Xt)["buy"])}
    print(f"=== CANDIDATE-ROW AUC on TEST ({len(yl_t):,} rows, {int(yl_t.sum())} love, {int(yb_t.sum())} buy) ===")
    print(pd.DataFrame(rows).T, "\n")

    # ---------- top-10 quality ----------
    models = [ColdUserFallback(pop_model, data)]
    for name, rk in rankers.items():
        ts = TwoStageRec(gen, fb, rk, ev_test.history)
        ts.name = name
        models.append(ColdUserFallback(ts, data))

    for m in models[1:]:
        t0 = time.perf_counter()
        m.score(users[:500])
        print(f"{m.name:<22} inference: {(time.perf_counter() - t0) / 500 * 1000:.2f} ms/user")
    print()

    results = {m.name: ev_test.evaluate(m) for m in models}
    for seg in ("all", "warm", "cold"):
        print(f"=== {seg.upper()} users, K=10 (TEST) ===")
        print(leaderboard(results, k=10, segment=seg), "\n")

    rec, total = cold_item_recall(data, models)
    print(f"=== COLD-ITEM view: {total} test purchases of unseen items ===")
    print(rec.sort_values(ascending=False))

    PATHS.make()
    leaderboard(results, k=10).to_csv(PATHS.artifacts / "results_deep_ranker.csv")


if __name__ == "__main__":
    main()
