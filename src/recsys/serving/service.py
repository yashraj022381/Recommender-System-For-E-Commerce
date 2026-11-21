import time
import pickle
import threading

import numpy as np
import pandas as pd

from scipy import sparse as sp
from sklearn.preprocessing import normalize

from recsys.config import PATHS
from recsys.data.loader import load_dataset
from recsys.models.hybrid import HybridRec
from recsys.models.ranker import FeatureBuilder, TwoStageRec
from recsys.models.rules import FreshFavoriteRule
from recsys.models.sequential import build_sequences
from recsys.models.wrappers import ColdUserFallback
from recsys.retrieval.candidates import CandidateGenerator
from recsys.serving.business import BusinessReranker, simulate_stock


BUNDLE = PATHS.artifacts / "bundle.pkl"

MODEL_ORDER = ["Popularity", "UserKNN", "ItemKNN", "SVD", "ALS", "Content-TFIDF", "NCF", "TwoTower",
               "GRU4Rec", "SASRec", "LightGCN", "Generative", "TwoStage (rank)", "Hybrid blend",
               "Rule: fresh x favorite", "Final: Generative + shelf rules"]

# Phase 12 showed that blending does not beat Generative alone; we keep a small blend so that
# the "Why this?" box can show how each helper contributed.
BLEND = {"Generative": 1.0, "UserKNN": 0.25, "TwoTower": 0.25, "Content": 0.1}
# models without their own explanation borrow the closest model that has one
EXPLAIN_FROM = {"TwoStage (rank)": "TwoTower", "Final: Generative + shelf rules": "Generative",
                "Popularity": None, "Rule: fresh x favorite": None}


class RecService:
    SEQUENTIAL = ("SASRec", "GRU4Rec", "Generative")
    SIMILAR_METHODS = ("Co-purchase", "Description (TF-IDF)", "Two-Tower codes")

    def __init__(self, bundle_path=BUNDLE):
        self._lock = threading.RLock()       # Streamlit may serve several sessions at once
        self.data = data = load_dataset()
        self.n_users, self.n_items = data.n_users, data.n_items
        if self.n_items > 8000:
            raise RuntimeError(f"{self.n_items:,} items: the dense item-item tables would need about "
                               f"{3 * self.n_items ** 2 * 4 / 1e9:.1f} GB. Raise RECSYS_MIN_ITEM and re-run prepare_data.py.")
        with open(bundle_path, "rb") as f:   # a local file we wrote ourselves; never unpickle untrusted files
            bundle = pickle.load(f)
        self.raw = raw = bundle["parts"]

        # ---- "the story so far" = train + validation, as the Evaluator uses for the test split ----
        self.ctx = pd.concat([data.train, data.val], ignore_index=True)
        self.history = (data.train_csr + data.csr(data.val)).tocsr()
        self.seqs = build_sequences(self.ctx, self.n_users)
        R = data.train_pos_csr
        self.has_history = np.asarray(R.sum(1)).ravel() > 0

        # ---- item tables ----
        items = data.items
        cat = items["category"].astype("category")
        self.cats = cat.cat.codes.to_numpy()
        self.cat_labels = [str(c) for c in cat.cat.categories]
        self.cats_name = np.array([self.cat_labels[c] for c in self.cats])
        self.titles = items["title"].fillna("").tolist()
        self.price = items["price"].to_numpy(dtype=float)
        self.item_pop = np.asarray(R.sum(0)).ravel()
        self.fresh = np.asarray(data.train_csr.sum(0)).ravel() == 0
        self.oos = simulate_stock(self.n_items)              # SIMULATED stock flags

        # ---- user tables ----
        P = data.csr(self.ctx, positives_only=True)
        C = sp.csr_matrix((np.ones(len(self.cats)), (np.arange(len(self.cats)), self.cats)),
                          shape=(len(self.cats), int(self.cats.max()) + 1))
        A = (P @ C).toarray()
        self.aff = A / np.maximum(A.sum(1, keepdims=True), 1)       # share of a user's likes per category
        self.n_liked = np.asarray(P.sum(1)).ravel()
        self.user_price = (P @ self.price) / np.maximum(self.n_liked, 1)
        tp = data.test[data.test["label"] == 1]
        self.future = tp.groupby("user_id")["item_id"].apply(list).to_dict()   # what they really bought next
        self.demo_users = sorted(self.future)
        self.cold_users = [u for u in self.demo_users if not self.has_history[u]]

        # ---- item-to-item similarity (dense: fine for ~1k items) ----
        X = raw["Content-TFIDF"].X
        self.sim_content = (X @ X.T).toarray().astype(np.float32)
        Rn = normalize(R.T.tocsr().astype(np.float32))
        self.sim_co = (Rn @ Rn.T).toarray().astype(np.float32)
        codes = raw["TwoTower"].item_codes
        self.sim_tt = (codes @ codes.T).astype(np.float32)

        # ---- assemble the 16 models (all behind the same cold-user rule) ----
        wrap = lambda m: ColdUserFallback(m, data)
        tt, content, iknn = raw["TwoTower"], raw["Content-TFIDF"], raw["ItemKNN"]
        cg = CandidateGenerator(tt, content, iknn, self.item_pop)
        fb = FeatureBuilder(data, tt, content, iknn)
        two_stage = TwoStageRec(cg, fb, bundle["ranker"], self.history)
        hybrid = HybridRec({"Generative": raw["Generative"], "UserKNN": raw["UserKNN"],
                            "TwoTower": tt, "Content": content}, BLEND, data)
        final = BusinessReranker(wrap(raw["Generative"]), data, self.sim_content, self.cats,
                                 self.fresh, self.oos, lam=0.7, cat_cap=5, fresh_slots=2,
                                 hide_oos=True)
        models = {name: wrap(m) for name, m in raw.items()}
        models.update({"TwoStage (rank)": wrap(two_stage), "Hybrid blend": wrap(hybrid),
                       "Rule: fresh x favorite": wrap(FreshFavoriteRule(data)),
                       "Final: Generative + shelf rules": final})
        self.models = {n: models[n] for n in MODEL_ORDER}
        for n, m in self.models.items():
            m.name = n                                       # display names, used by the evaluators
        for m in self.models.values():
            if hasattr(m, "set_context"):
                m.set_context(self.ctx)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _top(s, k):
        top = np.argpartition(-s, k - 1)[:k]
        return top[np.argsort(-s[top])]

    def _cards(self, items, scores=None):
        out = []
        for r, i in enumerate(items):
            i = int(i)
            out.append({"rank": r + 1, "item_id": i, "title": self.titles[i] or f"Item {i}",
                        "category": str(self.cats_name[i]), "price": float(self.price[i]),
                        "score": None if scores is None else float(scores[r]),
                        "new": bool(self.fresh[i]), "liked_by": int(self.item_pop[i]),
                        "oos": bool(self.oos[i])})
        return out

    # ------------------------------------------------------------------ requests
    def recommend(self, model_name, user_id, k=10):
        """-> (cards, milliseconds). Items the user already owns are never shown. k <= 20."""
        t0 = time.perf_counter()
        with self._lock:
            s = np.array(self.models[model_name].score(np.array([user_id])),
                         dtype=np.float32, copy=True)[0]
        s[self.history[user_id].indices] = -np.inf
        top = self._top(s, k)
        return self._cards(top, s[top]), (time.perf_counter() - t0) * 1000

    def similar_items(self, item_id, method="Co-purchase", k=10):
        M = {"Co-purchase": self.sim_co, "Description (TF-IDF)": self.sim_content,
             "Two-Tower codes": self.sim_tt}[method]
        s = M[item_id].astype(np.float32)               # astype copies
        s[item_id] = -np.inf
        top = self._top(s, k)
        if method == "Co-purchase":
            top = top[s[top] > 0]          # new items have no co-purchases: show nothing, not noise
        return self._cards(top, s[top])

    def search(self, query, k=10):
        """Keyword search over item descriptions (TF-IDF). Returns [] if no word matches."""
        c = self.raw["Content-TFIDF"]
        s = (c.X @ c.vec.transform([query]).T).toarray().ravel().astype(np.float32)
        if s.size == 0 or s.max() <= 0:
            return []
        top = self._top(s, min(k, len(s)))
        top = top[s[top] > 0]
        return self._cards(top, s[top])

    def from_items(self, liked, k=10, exclude=()):
        """Cold-start onboarding: a brand-new shopper names a few products they like."""
        liked = [int(i) for i in liked]
        if liked:
            a, b = self.sim_co[liked].mean(0), self.sim_content[liked].mean(0)
            s = a / (a.max() + 1e-9) + b / (b.max() + 1e-9) \
                + 1e-3 * self.item_pop / (self.item_pop.max() + 1e-9)
        else:
            s = self.item_pop.astype(np.float32)
        s = np.array(s, dtype=np.float32)
        s[liked] = -np.inf
        for i in exclude:
            s[int(i)] = -np.inf
        top = self._top(s, k)
        return self._cards(top, s[top])

    def live_recommend(self, model_name, user_id, session_items, k=10):
        """Recommendations right after the shopper clicked `session_items` (oldest first).
        -> (cards, mode description, milliseconds)"""
        t0 = time.perf_counter()
        session = [int(i) for i in session_items]
        if not session:
            cards, _ = self.recommend(model_name, user_id, k)
            mode = "no new clicks yet: showing the normal list"
        elif model_name in self.SEQUENTIAL:
            base = self.raw[model_name]                  # unwrapped, so a new shopper's clicks count too
            with self._lock:
                orig = base.seqs[user_id]
                base.seqs[user_id] = np.concatenate([orig, np.asarray(session, dtype=np.int64)])
                try:
                    s = np.array(base.score(np.array([user_id])), dtype=np.float32, copy=True)[0]
                finally:
                    base.seqs[user_id] = orig            # never leave the shared model changed
            s[self.history[user_id].indices] = -np.inf
            s[session] = -np.inf
            top = self._top(s, k)
            cards = self._cards(top, s[top])
            mode = "the model re-read the story, including the new clicks"
        else:
            cards = self.from_items(session, k, exclude=self.history[user_id].indices)
            mode = "this model cannot read live clicks: toy-to-toy matching on the new clicks"
        return cards, mode, (time.perf_counter() - t0) * 1000

    def user_profile(self, user_id):
        top_cats = sorted(((self.cat_labels[c], float(v)) for c, v in enumerate(self.aff[user_id])
                           if v > 0), key=lambda p: -p[1])[:3]
        return {"cold": not bool(self.has_history[user_id]), "n_liked": int(self.n_liked[user_id]),
                "top_categories": top_cats, "recent": self._cards(self.seqs[user_id][-5:][::-1]),
                "future": self.future.get(user_id, [])}

    # ------------------------------------------------------------------ "Why this?"
    def _evidence_lines(self, src, ev):
        if not ev:
            return []
        if src == "UserKNN":
            who = ", ".join(f"shopper #{u} (similarity {s:.2f})" for u, s in ev)
            return [f"Shoppers with a similar taste liked it: {who}."]
        if src == "Generative":
            hits = [(i, n) for i, n in ev if n > 0]
            if not hits:
                return []
            what = ", ".join(f"'{self.titles[i]}' ({n} shared tokens)" for i, n in hits)
            return [f"Its content address overlaps with your recent likes: {what}."]
        if src == "Hybrid blend":
            return ["Blend of helpers: " + ", ".join(f"{p} {s:.0%}" for p, s in ev) + "."]
        what = ", ".join(f"'{self.titles[i]}' ({s:.2f})" for i, s in ev)
        return [f"Similar to items you liked: {what}."]

    def explain(self, model_name, user_id, item_id):
        """-> {'lines': [plain-English reasons]}. Never empty."""
        lines = []
        src = EXPLAIN_FROM.get(model_name, model_name)
        m = self.models.get(src) if src else None
        if not self.has_history[user_id]:
            lines.append("New shopper with no history, so the list falls back to overall popularity.")
        elif m is not None and hasattr(m, "explain"):
            try:
                lines += self._evidence_lines(src, m.explain(user_id, item_id))
                if hasattr(m, "top_terms"):
                    terms = [w for w, _ in m.top_terms(user_id, item_id)]
                    if terms:
                        lines.append("Matching words in the descriptions: " + ", ".join(terms) + ".")
            except Exception:                    # an explanation must never break the page
                pass
        share = float(self.aff[user_id, self.cats[item_id]])
        if share > 0:
            lines.append(f"{share:.0%} of your liked items are in '{self.cats_name[item_id]}'.")
        if self.fresh[item_id]:
            lines.append("Brand-new item with no purchase history yet; it can only be matched "
                         "through its description.")
            if model_name.startswith("Final"):
                lines.append("The shelf rules reserve slots for new arrivals.")
        else:
            lines.append(f"Liked by {int(self.item_pop[item_id])} shoppers so far.")
        if self.n_liked[user_id] > 0:
            lines.append(f"Price ${self.price[item_id]:.2f} vs your usual ${self.user_price[user_id]:.2f}.")
        return {"lines": lines}




        
