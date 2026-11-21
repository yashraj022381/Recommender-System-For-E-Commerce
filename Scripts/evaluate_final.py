"""Run:  python scripts/evaluate_final.py     (about 10-15 minutes on CPU)"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.beyond import beyond_accuracy
from recsys.evaluation.cold import cold_item_recall
from recsys.evaluation.evaluator import Evaluator, summary
from recsys.evaluation.stats import paired_bootstrap, run_seeds
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.generative import GenerativeRec
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.neural import TwoTowerRec
from recsys.models.ranker import FeatureBuilder, TwoStageRec
from recsys.models.rules import FreshFavoriteRule
from recsys.models.sequential import SASRecRec
from recsys.models.wrappers import ColdUserFallback
from recsys.retrieval.candidates import CandidateGenerator

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)
SEEDS = (0, 1, 2)


class ShuffledAddresses(GenerativeRec):
    """CONTROL: same model, but item addresses are randomly permuted (no text information)."""

    def fit(self, data):
        import recsys.models.generative as g
        orig = g.semantic_ids
        rng = np.random.default_rng(self.seed)

        def scrambled(vectors, levels, codebook, seed):
            c = orig(vectors, levels, codebook, seed)
            return c[rng.permutation(len(c))]

        g.semantic_ids = scrambled
        try:
            return super().fit(data)
        finally:
            g.semantic_ids = orig


class ShuffledSASRec(SASRecRec):
    """CONTROL: each user's history order is randomly shuffled at train AND test time."""

    def set_context(self, df):
        super().set_context(df)
        rng = np.random.default_rng(123)
        self.seqs = [rng.permutation(s) for s in self.seqs]


class _Trim:
    def __init__(self, model, drop):
        self.model, self.drop = model, drop

    def predict_proba(self, X):
        return self.model.predict_proba(np.delete(X, self.drop, axis=1))


def main():
    data = load_dataset()
    ev_val = Evaluator(data, split="val", ks=(10,))
    ev = Evaluator(data, split="test", ks=(10,))
    wrap = lambda m: ColdUserFallback(m, data)

    # ---------- 1. models, 3 seeds each (per-user scores averaged over seeds) ----------
    print("Training seeded models (this is the slow part)...")
    per_user, seed_scores = {}, {}
    specs = {
        "Generative": lambda s: GenerativeRec(dim=64, layers=1, epochs=40, seed=s),
        "Generative-shuffledIDs": lambda s: ShuffledAddresses(dim=64, layers=1, epochs=40, seed=s),
        "SASRec": lambda s: SASRecRec(dim=32, layers=1, epochs=40, seed=s),
        "SASRec-shuffledOrder": lambda s: ShuffledSASRec(dim=32, layers=1, epochs=40, seed=s),
        "TwoTower": lambda s: TwoTowerRec(**{**BEST_TT, "seed": s}),
    }
    for name, make in specs.items():
        per_user[name], seed_scores[name] = run_seeds(make, data, ev, SEEDS, wrap=wrap)
        print(f"  {name:<24} NDCG@10 by seed: {[round(v, 4) for v in seed_scores[name]]}")

    # deterministic models: one run is enough
    uknn = UserKNN(k=400).fit(data)
    for name, m in (("UserKNN", uknn), ("Popularity", PopularityRec().fit(data)),
                    ("Rule: fresh x favorite", FreshFavoriteRule(data))):
        per_user[name] = ev.evaluate(wrap(m))["ndcg@10"]

    # ---------- 2. two-stage ranker, with and without item_pop ----------
    iknn, content = ItemKNN(k=400).fit(data), TfidfContentRec().fit(data)
    tt = TwoTowerRec(**BEST_TT).fit(data)
    pop = np.asarray(data.train_pos_csr.sum(0)).ravel()
    gen = CandidateGenerator(tt, content, iknn, pop)
    fb = FeatureBuilder(data, tt, content, iknn)
    has_hist = np.asarray(data.train_pos_csr.sum(1)).ravel() > 0
    tr_users = np.flatnonzero(has_hist)
    m_val = gen.generate(tr_users, ev_val.history)["union"]
    X, r, c = fb.build(tr_users, m_val, ev_val.history)
    y = np.asarray(ev_val.truth[tr_users[r], c]).ravel()
    names = FeatureBuilder.NAMES
    for label, drop in (("TwoStage-GBDT", []), ("TwoStage-noItemPop", [names.index("item_pop")])):
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                            random_state=0).fit(np.delete(X, drop, axis=1), y)
        ts = TwoStageRec(gen, fb, _Trim(gb, drop), ev.history)
        ts.name = label
        per_user[label] = ev.evaluate(wrap(ts))["ndcg@10"]

    # ---------- 3. table + paired tests ----------
    base = "UserKNN"
    rows = []
    for name, s in per_user.items():
        d, lo, hi, p = paired_bootstrap(s, per_user[base]) if name != base else (0, 0, 0, 1)
        rows.append({"model": name, "NDCG@10": s.mean(), "diff vs UserKNN": d,
                     "95% CI low": lo, "95% CI high": hi, "p": p})
    table = pd.DataFrame(rows).set_index("model").sort_values("NDCG@10", ascending=False)
    print("\n=== NDCG@10 (TEST), paired bootstrap vs UserKNN ===")
    print(table, "\n")

    print("=== CONTROLS (paired, 3-seed averages) ===")
    for a, b, what in (("Generative", "Generative-shuffledIDs", "text addresses matter?"),
                       ("SASRec", "SASRec-shuffledOrder", "history order matters?"),
                       ("Generative", "SASRec", "Generative vs SASRec"),
                       ("TwoStage-GBDT", "TwoStage-noItemPop", "item_pop matters?"),
                       ("TwoStage-GBDT", "Rule: fresh x favorite", "ranker vs one-line rule")):
        d, lo, hi, p = paired_bootstrap(per_user[a], per_user[b])
        print(f"  {what:<26} {a} - {b}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")

    # ---------- 4. diversity / novelty ----------
    X_c = content.X
    item_sim = (X_c @ X_c.T).toarray().astype(np.float32)
    cats = data.items["category"].astype("category").cat.codes.to_numpy()
    train_pop = np.asarray(data.train_pos_csr.sum(0)).ravel().astype(np.float64)
    shelf = {"Popularity": wrap(PopularityRec().fit(data)), "UserKNN": wrap(uknn),
             "Generative": wrap(GenerativeRec(dim=64, layers=1, epochs=40).fit(data)),
             "Rule: fresh x favorite": wrap(FreshFavoriteRule(data))}
    print("\n=== BEYOND ACCURACY (top-10 lists, TEST) ===")
    print(pd.DataFrame({k: beyond_accuracy(m, ev, train_pop, item_sim, cats) for k, m in shelf.items()}).T)

    PATHS.make()
    table.to_csv(PATHS.artifacts / "results_final.csv")


if __name__ == "__main__":
    main()
