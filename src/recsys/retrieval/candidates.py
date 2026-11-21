"""Stage 1: pool the top picks of several fast 'helpers'."""
import numpy as np

from recsys.retrieval.index import VectorIndex

SOURCES = ("tt", "content", "iknn", "pop")


class CandidateGenerator:
    """Returns boolean masks (n_users, n_items). At real scale you would return id lists instead;
    masks keep this project simple and fast for ~1k items."""

    def __init__(self, tt, content, iknn, pop, n_tt=100, n_content=50, n_iknn=50, n_pop=30):
        self.tt, self.content, self.iknn = tt, content, iknn
        self.pop = np.asarray(pop, dtype=np.float32)
        self.n = dict(tt=n_tt, content=n_content, iknn=n_iknn, pop=n_pop)
        self.index = VectorIndex(tt.item_codes)

    @staticmethod
    def _topn_mask(scores, n):
        n = min(n, scores.shape[1])
        top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
        m = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(m, top, True, axis=1)
        return m

    def generate(self, users, history):
        users = np.asarray(users)
        seen = history[users].toarray() > 0
        out = {}

        # Two-Tower via ANN: over-fetch, then drop items the user already owns
        q = self.tt.user_codes(users)
        _, I = self.index.search(q, self.n["tt"] + int(seen.sum(1).max()))
        m = np.zeros_like(seen)
        for r in range(len(users)):
            keep = I[r][~seen[r, I[r]]][: self.n["tt"]]
            m[r, keep] = True
        out["tt"] = m

        for name, model in (("content", self.content), ("iknn", self.iknn)):
            s = np.array(model.score(users), dtype=np.float32, copy=True)
            s[seen] = -np.inf
            out[name] = self._topn_mask(s, self.n[name])

        p = np.broadcast_to(self.pop, seen.shape).copy()
        p[seen] = -np.inf
        out["pop"] = self._topn_mask(p, self.n["pop"])

        for k in SOURCES:
            out[k] &= ~seen
        out["union"] = np.logical_or.reduce([out[k] for k in SOURCES])
        return out
