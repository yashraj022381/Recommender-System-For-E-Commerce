"""Neural recommenders (PyTorch): NCF and a content-aware Two-Tower model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.data.loader import Dataset
from recsys.models.base import Recommender


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Neural(Recommender):
    """Shared plumbing: positive pairs, tie-breaker, training loop skeleton."""

    def _prepare(self, data: Dataset):
        self.n_users, self.n_items = data.n_users, data.n_items
        self.R = data.train_pos_csr.astype(np.float32)
        pos = data.train[data.train["label"] == 1]
        self.u_pos = pos["user_id"].to_numpy(dtype=np.int64)
        self.i_pos = pos["item_id"].to_numpy(dtype=np.int64)
        pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)
        self.has_history = np.asarray(self.R.sum(1)).ravel() > 0

    def _batches(self, rng, batch_size):
        order = rng.permutation(len(self.u_pos))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            yield self.u_pos[idx], self.i_pos[idx]

    def _sample_negatives(self, rng, users, n_neg):
        """Random items; a few may be true positives (acceptable noise, standard practice)."""
        return rng.integers(0, self.n_items, size=(len(users), n_neg))

    def _cold_user_fallback(self, scores, user_ids):
        """Users with no training history get popularity scores (tiebreak * 1e6 is popularity in [0, 1])."""
        cold = ~self.has_history[np.asarray(user_ids)]
        if cold.any():
            scores[cold] = self.tiebreak * 1e6
        return scores


# --------------------------------------------------------------------------- NCF
class _NCFNet(nn.Module):
    """GMF branch (element-wise product) + MLP branch, fused (He et al., 2017)."""

    def __init__(self, n_users, n_items, dim, hidden):
        super().__init__()
        self.ug, self.ig = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.um, self.im = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.out = nn.Linear(dim + hidden // 2, 1)
        for e in (self.ug, self.ig, self.um, self.im):
            nn.init.normal_(e.weight, std=0.05)

    def forward(self, u, i):
        g = self.ug(u) * self.ig(i)
        m = self.mlp(torch.cat([self.um(u), self.im(i)], -1))
        return self.out(torch.cat([g, m], -1)).squeeze(-1)


class NCFRec(_Neural):
    def __init__(self, dim=32, hidden=64, epochs=12, lr=2e-3, n_neg=4,
                 batch_size=1024, weight_decay=1e-6, seed=0):
        self.dim, self.hidden, self.epochs, self.lr = dim, hidden, epochs, lr
        self.n_neg, self.batch_size, self.wd, self.seed = n_neg, batch_size, weight_decay, seed
        self.name = f"NCF(d={dim})"

    def fit(self, data: Dataset):
        set_seed(self.seed)
        self._prepare(data)
        rng = np.random.default_rng(self.seed)
        self.net = _NCFNet(self.n_users, self.n_items, self.dim, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            self.net.train()
            for u, i in self._batches(rng, self.batch_size):
                neg = self._sample_negatives(rng, u, self.n_neg)
                uu = torch.from_numpy(np.concatenate([u, np.repeat(u, self.n_neg)]))
                ii = torch.from_numpy(np.concatenate([i, neg.ravel()]))
                y = torch.cat([torch.ones(len(u)), torch.zeros(len(u) * self.n_neg)])
                loss = F.binary_cross_entropy_with_logits(self.net(uu, ii), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, user_ids):
        u = torch.as_tensor(np.asarray(user_ids), dtype=torch.long)
        out = np.empty((len(u), self.n_items), dtype=np.float32)
        all_i = torch.arange(self.n_items)
        for r in range(len(u)):                      # one user at a time keeps memory small
            out[r] = self.net(u[r].expand(self.n_items), all_i).numpy()
        return out + self.tiebreak

    @torch.no_grad()
    def user_codes(self, user_ids):
        u = torch.as_tensor(np.asarray(user_ids), dtype=torch.long)
        return self.net.user_vec(u).numpy()


# ----------------------------------------------------------------------- Two-Tower
def build_item_features(data: Dataset, text_dim: int = 64, n_price_buckets: int = 8):
    """Per item: reduced TF-IDF vector, category id, price bucket."""
    items = data.items
    text = (items["title"].fillna("") + " " + items["text"].fillna("")).tolist()
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(text)
    k = min(text_dim, tfidf.shape[1] - 1)
    dense = TruncatedSVD(n_components=k, random_state=0).fit_transform(tfidf).astype(np.float32)
    dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
    cat = items["category"].astype("category").cat.codes.to_numpy().astype(np.int64)
    price = items["price"].to_numpy(dtype=np.float64)
    edges = np.quantile(price, np.linspace(0, 1, n_price_buckets + 1)[1:-1])
    bucket = np.digitize(price, edges).astype(np.int64)
    return dense, cat, bucket, int(cat.max()) + 1, n_price_buckets


class _TwoTowerNet(nn.Module):
    def __init__(self, n_users, n_items, n_cats, n_buckets, text_dim, dim, id_dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.user_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim))
        self.item_emb = nn.Embedding(n_items, dim)
        self.cat_emb = nn.Embedding(n_cats, 16)
        self.price_emb = nn.Embedding(n_buckets, 8)
        self.text_proj = nn.Linear(text_dim, 32)
        self.item_mlp = nn.Sequential(nn.Linear(dim + 16 + 8 + 32, 64), nn.ReLU(),
                                      nn.Linear(64, dim))
        self.id_dropout = id_dropout
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def user_vec(self, u):
        return F.normalize(self.user_mlp(self.user_emb(u)), dim=-1)

    def item_vec(self, i, text, cat, bucket):
        e = self.item_emb(i)
        if self.training and self.id_dropout > 0:    # teach the tower to work WITHOUT the ID,
            keep = (torch.rand(e.shape[0], 1) > self.id_dropout).float()   # so cold items work
            e = e * keep
        x = torch.cat([e, self.cat_emb(cat), self.price_emb(bucket),
                       F.relu(self.text_proj(text))], -1)
        return F.normalize(self.item_mlp(x), dim=-1)


class TwoTowerRec(_Neural):
    """Dual encoder trained with in-batch softmax. Cold items are scored from content alone."""

    def __init__(self, dim=32, text_dim=64, epochs=15, lr=3e-3, batch_size=512,
                 temperature=0.1, id_dropout=0.3, weight_decay=1e-6, seed=0, logq=True):
        self.dim, self.text_dim, self.epochs, self.lr = dim, text_dim, epochs, lr
        self.batch_size, self.temp, self.id_dropout = batch_size, temperature, id_dropout
        self.wd, self.seed = weight_decay, seed
        self.logq = logq
        self.name = f"TwoTower(d={dim})"

    def fit(self, data: Dataset):
        set_seed(self.seed)
        self._prepare(data)
        rng = np.random.default_rng(self.seed)
        text, cat, bucket, n_cats, n_buckets = build_item_features(data, self.text_dim)
        self.t_text = torch.from_numpy(text)
        self.t_cat, self.t_bucket = torch.from_numpy(cat), torch.from_numpy(bucket)
        self.net = _TwoTowerNet(self.n_users, self.n_items, n_cats, n_buckets,
                                text.shape[1], self.dim, self.id_dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        item_ids = torch.arange(self.n_items)
        cnt = np.bincount(self.i_pos, minlength=self.n_items).astype(np.float64)
        self.log_q = torch.from_numpy(np.log(np.maximum(cnt / cnt.sum(), 1e-9))).float()

        for _ in range(self.epochs):
            self.net.train()
            for u, i in self._batches(rng, self.batch_size):
                u_t, i_t = torch.from_numpy(u), torch.from_numpy(i)
                uv = self.net.user_vec(u_t)
                iv = self.net.item_vec(i_t, self.t_text[i_t], self.t_cat[i_t], self.t_bucket[i_t])
                logits = uv @ iv.T / self.temp                # (B, B): diagonal = true pairs
                if self.logq:                                       # undo popularity bias of in-batch negatives
                    logits = logits - self.log_q[i_t][None, :]
                # a user's other liked items in the batch are not real negatives -> mask them
                same_user = torch.from_numpy(u[:, None] == u[None, :])
                same_item = torch.from_numpy(i[:, None] == i[None, :])
                mask = (same_user | same_item) & ~torch.eye(len(u), dtype=torch.bool)
                logits = logits.masked_fill(mask, -1e9)
                loss = F.cross_entropy(logits, torch.arange(len(u)))
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.net.eval()
        with torch.no_grad():                                       # precompute all item codes once
            self.item_codes = self.net.item_vec(item_ids, self.t_text, self.t_cat,
                                                self.t_bucket).numpy()
        self._seen_items = np.asarray(self.R.sum(0)).ravel() > 0
        return self

    @torch.no_grad()
    def score(self, user_ids):
        user_ids = np.asarray(user_ids)
        u = torch.as_tensor(user_ids, dtype=torch.long)
        #uv = self.net.user_vec(u).numpy()
        out = self.net.user_vec(u).numpy() @ self.item_codes.T + self.tiebreak
        #return uv @ self.item_codes.T + self.tiebreak
        return self._cold_user_fallback(out, user_ids)

    @torch.no_grad()
    def user_codes(self, user_ids):
        """User vectors for ANN search."""
        u = torch.as_tensor(np.asarray(user_ids), dtype=torch.long)
        return self.net.user_vec(u).numpy()

    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Liked items whose tower code is closest to the recommended item's code."""
        liked = self.R[user_id].indices
        if len(liked) == 0:
            return []
        sims = self.item_codes[liked] @ self.item_codes[item_id]
        order = np.argsort(-sims)[:n]
        return [(int(liked[j]), float(sims[j])) for j in order]
