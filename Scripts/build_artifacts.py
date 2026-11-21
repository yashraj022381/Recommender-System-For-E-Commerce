"""Run:  python scripts/build_artifacts.py     (about 4-6 minutes on CPU)"""
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.evaluator import Evaluator
from recsys.models.baselines import PopularityRec
from recsys.models.content import TfidfContentRec
from recsys.models.generative import GenerativeRec
from recsys.models.gnn import LightGCNRec
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.mf import ALSRec, SVDRec
from recsys.models.neural import NCFRec, TwoTowerRec
from recsys.models.ranker import FeatureBuilder
from recsys.models.sequential import GRU4RecRec, SASRecRec
from recsys.retrieval.candidates import CandidateGenerator
from recsys.serving.service import BUNDLE

BEST_TT = dict(dim=16, id_dropout=0.3, temperature=0.2, epochs=8, logq=True)

SPECS = {   # the best configs found on the validation split in earlier phases
    "Popularity": lambda: PopularityRec(),
    "UserKNN": lambda: UserKNN(k=400),
    "ItemKNN": lambda: ItemKNN(k=400),
    "SVD": lambda: SVDRec(factors=4),
    "ALS": lambda: ALSRec(factors=8, alpha=10.0, reg=1.0),
    "Content-TFIDF": lambda: TfidfContentRec(),
    "NCF": lambda: NCFRec(dim=32, epochs=10),
    "TwoTower": lambda: TwoTowerRec(**BEST_TT),
    "GRU4Rec": lambda: GRU4RecRec(dim=64, epochs=20),
    "SASRec": lambda: SASRecRec(dim=32, layers=1, epochs=40),
    "LightGCN": lambda: LightGCNRec(dim=32, layers=3, epochs=40),
    "Generative": lambda: GenerativeRec(dim=64, layers=1, epochs=40),
}


def main():
    data = load_dataset()
    parts = {}
    for name, make in SPECS.items():
        t0 = time.time()
        parts[name] = make().fit(data)
        print(f"  fitted {name:<14} {time.time() - t0:5.1f}s")

    # the graph is only needed for training; dropping it keeps the pickle simple
    parts["LightGCN"].A = None

    # ---- the judge of the two-stage model: trained on VALIDATION labels (as in Phase 7) ----
    print("  training the two-stage ranker...")
    ev_val = Evaluator(data, split="val", ks=(10,))
    pop = np.asarray(data.train_pos_csr.sum(0)).ravel()
    gen = CandidateGenerator(parts["TwoTower"], parts["Content-TFIDF"], parts["ItemKNN"], pop)
    fb = FeatureBuilder(data, parts["TwoTower"], parts["Content-TFIDF"], parts["ItemKNN"])
    tr_users = np.flatnonzero(np.asarray(data.train_pos_csr.sum(1)).ravel() > 0)
    mask = gen.generate(tr_users, ev_val.history)["union"]
    X, r, c = fb.build(tr_users, mask, ev_val.history)
    y = np.asarray(ev_val.truth[tr_users[r], c]).ravel()
    ranker = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                            random_state=0).fit(X, y)

    PATHS.make()
    with open(BUNDLE, "wb") as f:
        pickle.dump({"parts": parts, "ranker": ranker}, f)
    print(f"\nSaved {BUNDLE}  ({BUNDLE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
