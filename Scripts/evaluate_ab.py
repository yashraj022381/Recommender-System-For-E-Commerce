"""Run:  python scripts/evaluate_ab.py     (about 3-5 minutes on CPU)   *** SIMULATION ***"""
import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.evaluation.abtest import (exposure, funnel, repeat_ab, revenue_lift, simulate,
                                      top_lists)
from recsys.evaluation.evaluator import Evaluator
from recsys.models.content import TfidfContentRec
from recsys.models.generative import GenerativeRec
from recsys.models.knn import UserKNN
from recsys.models.rules import FreshFavoriteRule
from recsys.models.wrappers import ColdUserFallback
from recsys.serving.business import BusinessReranker, simulate_stock

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)

SESSIONS = 10
PAIRS = [("UserKNN", "UserKNN"),                                   # A/A: should show nothing
         ("UserKNN", "Generative"),
         ("Generative", "Generative+shelf rules"),
         ("UserKNN", "Rule: fresh x favorite")]


def main():
    data = load_dataset()
    ev = Evaluator(data, split="test", ks=(10,))
    wrap = lambda m: ColdUserFallback(m, data)

    print("Fitting models...")
    gen = GenerativeRec(dim=64, layers=1, epochs=40).fit(data)
    uknn = UserKNN(k=400).fit(data)
    content = TfidfContentRec().fit(data)
    item_sim = (content.X @ content.X.T).toarray().astype(np.float32)
    cats = data.items["category"].astype("category").cat.codes.to_numpy()
    fresh = np.asarray(data.train_csr.sum(0)).ravel() == 0
    oos = simulate_stock(data.n_items)                              # SIMULATED stock flags
    price = data.items["price"].to_numpy(dtype=np.float64)

    shelf = BusinessReranker(wrap(gen), data, item_sim, cats, fresh, oos, lam=0.7, cat_cap=5,
                             fresh_slots=2, hide_oos=True, name="Generative+shelf rules")
    systems = {"UserKNN": wrap(uknn), "Generative": wrap(gen), "Generative+shelf rules": shelf,
               "Rule: fresh x favorite": wrap(FreshFavoriteRule(data))}
    exps = {name: exposure(top_lists(m, ev), ev, price, oos) for name, m in systems.items()}

    print(f"\n*** SIMULATION: {len(ev.users):,} users, {SESSIONS} sessions each, "
          f"{int(oos.sum())} out-of-stock items. All click/cart/buy rates are ASSUMED. ***\n")
    summary = []
    for a, b in PAIRS:
        rng = np.random.default_rng(0)
        arm = rng.random(len(ev.users)) < 0.5
        da = simulate(exps[a], np.flatnonzero(arm), rng, SESSIONS)
        db = simulate(exps[b], np.flatnonzero(~arm), rng, SESSIONS)
        tab = pd.DataFrame({f"A: {a}": funnel(da, SESSIONS), f"B: {b}": funnel(db, SESSIONS)}).T
        lift, (lo, hi) = revenue_lift(da, db)
        any_sig, b_better = repeat_ab(exps[a], exps[b], reps=200, sessions=SESSIONS)
        print(f"--- {a}  ->  {b} ---")
        print(tab)
        print(f"revenue/user lift: {lift:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]   "
              f"| over 200 repeats: significant {any_sig:.0%} of the time, "
              f"significant AND B better {b_better:.0%}\n")
        summary.append({"A": a, "B": b, "lift": lift, "ci_low": lo, "ci_high": hi,
                        "sig_share": any_sig, "sig_B_better": b_better})

    PATHS.make()
    pd.DataFrame(summary).to_csv(PATHS.artifacts / "results_ab_simulated.csv", index=False)


if __name__ == "__main__":
    main()
