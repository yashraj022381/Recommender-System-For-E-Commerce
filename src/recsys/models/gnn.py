"""LightGCN (He et al., 2020) in plain PyTorch: no extra graph library needed."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.mf import latent_explain
from recsys.models.neural import set_seed


def normalized_adjacency(R: sp.csr_matrix) -> torch.Tensor:
    """Symmetric-normalized adjacency of the bipartite user-item graph, as a sparse torch tensor.

    A = [[0, R], [R^T, 0]],   A_hat = D^-1/2 A D^-1/2
    """
    n_u, n_i = R.shape
    A = sp.bmat([[None, R], [R.T, None]], format="coo", dtype=np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    d = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    A = sp.diags(d) @ A @ sp.diags(d)
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(A.data.astype(np.float32)),
                                   size=(n_u + n_i, n_u + n_i)).coalesce()


class LightGCNRec(Recommender):
    def __init__(self, dim=32, layers=2, epochs=60, lr=5e-3, batch_size=2048,
                 reg=1e-4, seed=0):
        self.dim, self.layers, self.epochs, self.lr = dim, layers, epochs, lr
        self.batch_size, self.reg, self.seed = batch_size, reg, seed
        self.name = f"LightGCN(d={dim},L={layers})"

    def _propagate(self):
        """Average the embeddings over 0..L rounds of neighbor-averaging."""
        x = self.emb.weight
        out = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.A, x)
            out.append(x)
        z = torch.stack(out).mean(0)
        return z[:self.n_users], z[self.n_users:]

    def fit(self, data: Dataset):
        set_seed(self.seed)
        self.n_users, self.n_items = data.n_users, data.n_items
        self.R = data.train_pos_csr.astype(np.float32)
        self.A = normalized_adjacency(self.R)
        pos = data.train[data.train["label"] == 1]
        u_pos = pos["user_id"].to_numpy(dtype=np.int64)
        i_pos = pos["item_id"].to_numpy(dtype=np.int64)
        pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)

        self.emb = nn.Embedding(self.n_users + self.n_items, self.dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        opt = torch.optim.Adam(self.emb.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            order = rng.permutation(len(u_pos))
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                u = torch.from_numpy(u_pos[idx])
                i = torch.from_numpy(i_pos[idx])
                j = torch.from_numpy(rng.integers(0, self.n_items, len(idx)))   # random negative
                zu, zi = self._propagate()                  # full-graph propagation (small graph)
                su = (zu[u] * zi[i]).sum(1)
                sj = (zu[u] * zi[j]).sum(1)
                bpr = -F.logsigmoid(su - sj).mean()
                e0 = self.emb.weight
                reg = self.reg * (e0[u].pow(2).sum() + e0[self.n_users + i].pow(2).sum()
                                  + e0[self.n_users + j].pow(2).sum()) / len(idx)
                loss = bpr + reg
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            zu, zi = self._propagate()
        self.user_factors = zu.numpy()
        self.item_factors = zi.numpy()
        return self

    def score(self, user_ids):
        return self.user_factors[np.asarray(user_ids)] @ self.item_factors.T + self.tiebreak

    def explain(self, user_id: int, item_id: int, n: int = 3):
        return latent_explain(self.item_factors, self.R[user_id].indices, item_id, n)
