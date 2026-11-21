"""Deep multitask ranker (PyTorch): MMoE experts + ESMM-style heads."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from recsys.models.neural import set_seed
from recsys.models.ranker import FeatureBuilder


class FeatureBuilderCat(FeatureBuilder):
    """Same 11 features as Phase 7, plus the item's category id as the LAST column."""

    def build(self, users, mask, history):
        X, r, c = super().build(users, mask, history)
        return np.column_stack([X, self.cat[c]]).astype(np.float32), r, c


class _MMoE(nn.Module):
    """Shared experts; each task has its own gate (which experts to trust) and tower."""

    def __init__(self, n_dense, n_cats, n_tasks, n_experts, hidden, dropout, cat_dim=8):
        super().__init__()
        self.cat_emb = nn.Embedding(n_cats, cat_dim)
        d_in = n_dense + cat_dim
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.ReLU())
            for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(d_in, n_experts) for _ in range(n_tasks)])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            for _ in range(n_tasks)])

    def forward(self, x, cat):
        h = torch.cat([x, self.cat_emb(cat)], -1)
        E = torch.stack([e(h) for e in self.experts], 1)              # (B, experts, hidden)
        outs = []
        for gate, tower in zip(self.gates, self.towers):
            w = F.softmax(gate(h), -1).unsqueeze(-1)                  # (B, experts, 1)
            outs.append(tower((w * E).sum(1)).squeeze(-1))            # one logit per task
        return outs


def _logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


class DeepRanker:
    """Input X: the 11 dense features + category id in the LAST column.

    multitask=True : heads  buy  and  cvr = P(love | buy);  score = P(buy) * P(cvr)  (ESMM)
    multitask=False: one head trained directly on the 'love' label
    Exposes predict_proba(X) so it plugs into TwoStageRec like the GBDT ranker.
    """

    def __init__(self, multitask=True, n_experts=3, hidden=64, epochs=8, lr=2e-3,
                 batch_size=2048, weight_decay=1e-5, dropout=0.1, seed=0):
        self.multitask, self.n_experts, self.hidden = multitask, n_experts, hidden
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.wd, self.dropout, self.seed = weight_decay, dropout, seed
        self.history = []                                   # held-out AUC per epoch (if monitored)

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=np.float32)
        return np.where(np.isfinite(X), X, 0.0)

    def _split(self, X):
        dense = (X[:, :-1] - self.mu) / self.sd
        cat = np.minimum(X[:, -1].astype(np.int64), self.n_cats - 1)
        return torch.from_numpy(dense.astype(np.float32)), torch.from_numpy(cat)

    def fit(self, X, y_buy, y_like, monitor=None):
        set_seed(self.seed)
        self.history = []
        X = self._clean(X)
        self.mu, self.sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-6
        self.n_cats = int(X[:, -1].max()) + 1
        xt, ct = self._split(X)
        yb = torch.from_numpy(np.asarray(y_buy, dtype=np.float32))
        yl = torch.from_numpy(np.asarray(y_like, dtype=np.float32))

        n_tasks = 2 if self.multitask else 1
        self.net = _MMoE(X.shape[1] - 1, self.n_cats, n_tasks, self.n_experts,
                         self.hidden, self.dropout)
        rate_b, rate_l = float(yb.mean()), float(yl.mean())     # start each head at its base rate
        if self.multitask:
            self.net.towers[0][-1].bias.data.fill_(_logit(rate_b))
            self.net.towers[1][-1].bias.data.fill_(_logit(rate_l / max(rate_b, 1e-4)))
        else:
            self.net.towers[0][-1].bias.data.fill_(_logit(rate_l))

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(xt))
            for s in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[s:s + self.batch_size])
                logits = self.net(xt[idx], ct[idx])
                if self.multitask:
                    p_like = torch.sigmoid(logits[0]) * torch.sigmoid(logits[1])
                    loss = (F.binary_cross_entropy_with_logits(logits[0], yb[idx])
                            + F.binary_cross_entropy(p_like.clamp(1e-7, 1 - 1e-7), yl[idx]))
                else:
                    loss = F.binary_cross_entropy_with_logits(logits[0], yl[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if monitor is not None:
                Xm, ym = monitor
                self.history.append(roc_auc_score(ym, self.predict_proba(Xm)[:, 1]))
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_tasks(self, X):
        """Returns {'buy': ..., 'like': ...}. 'buy' is NaN for the single-task model."""
        self.net.eval()
        xt, ct = self._split(self._clean(X))
        buy, like = [], []
        for s in range(0, len(xt), 65536):
            lg = self.net(xt[s:s + 65536], ct[s:s + 65536])
            if self.multitask:
                pb = torch.sigmoid(lg[0])
                pl = pb * torch.sigmoid(lg[1])
            else:
                pl = torch.sigmoid(lg[0])
                pb = torch.full_like(pl, float("nan"))
            buy.append(pb.numpy())
            like.append(pl.numpy())
        return {"buy": np.concatenate(buy), "like": np.concatenate(like)}

    def predict_proba(self, X):
        p = self.predict_tasks(X)["like"]
        return np.column_stack([1.0 - p, p])
