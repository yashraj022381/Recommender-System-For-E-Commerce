"""Generative recommender (TIGER-style, simplified).

Every item gets a short, unique 'semantic ID' (a few tokens) built from its TEXT.
A causal Transformer reads the user's history as semantic IDs, then writes the next item's
ID one token at a time. An item's score = log-probability of writing its ID.
Items never seen in training still have IDs (from content), so they can be scored.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.neural import build_item_features, set_seed
from recsys.models.sequential import _Block, build_sequences, left_pad, make_windows


def semantic_ids(vectors, levels: int = 3, codebook: int = 16, seed: int = 0) -> np.ndarray:
    """Residual k-means. Returns (n_items, levels + 1): `levels` cluster tokens plus a
    tie-break token, so every item has a UNIQUE ID."""
    res = np.asarray(vectors, dtype=np.float64).copy()
    codes = np.zeros((len(res), levels), dtype=np.int64)
    for l in range(levels):
        k = min(codebook, len(np.unique(res.round(5), axis=0)))
        if k > 1:
            km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(res)
            codes[:, l] = km.labels_
            res = res - km.cluster_centers_[km.labels_]        # next level clusters what is left over
    dup = pd.DataFrame(codes).groupby(list(range(levels))).cumcount().to_numpy()
    return np.column_stack([codes, dup])


class _GenNet(nn.Module):
    def __init__(self, sizes, dim, max_len, layers, dropout):
        super().__init__()
        self.sizes, self.n_lv = list(sizes), len(sizes)
        # input side: an item = sum of its token embeddings (index 0 = padding)
        self.in_emb = nn.ModuleList([nn.Embedding(s + 1, dim, padding_idx=0) for s in sizes])
        for e in self.in_emb:
            nn.init.normal_(e.weight, std=0.05)
            e.weight.data[0].zero_()
        # output side: tokens already written (used to condition the next token)
        self.prev_emb = nn.ModuleList([nn.Embedding(s, dim) for s in sizes[:-1]])
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([_Block(dim, 2, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, s)) for s in sizes])

    def hidden(self, x, codes_in):
        """x: (B, L) item ids + 1 (0 = pad).  codes_in: (n_items + 1, n_lv) tokens + 1, row 0 = pad."""
        c = codes_in[x]                                                   # (B, L, n_lv)
        e = self.drop(sum(emb(c[..., l]) for l, emb in enumerate(self.in_emb)))
        B, L = x.shape
        e = e + self.pos_emb(torch.arange(L))[None]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]    # look back only
        allow = causal & ~(x == 0)[:, None, :]                            # ...and not at padding
        allow = (allow | torch.eye(L, dtype=torch.bool)[None])[:, None]   # always see yourself
        for blk in self.blocks:
            e = blk(e, allow)
        return self.norm(e)

    def code_logits(self, h, level, prev_codes):
        """Distribution over the token at `level`, given the history vector and earlier tokens."""
        s = h
        for l, c in enumerate(prev_codes):
            s = s + self.prev_emb[l](c)
        return self.heads[level](s)

    @torch.no_grad()
    def item_logprobs(self, h, codes):
        """Exact log P(item's full ID | history) for EVERY item. h: (B, D); codes: (N, n_lv), 0-based."""
        B, N = h.shape[0], codes.shape[0]
        total = F.log_softmax(self.heads[0](h), -1)[:, codes[:, 0]]       # (B, N), level 0 needs no prefix
        for l in range(1, self.n_lv):
            prefix = sum(self.prev_emb[j](codes[:, j]) for j in range(l))  # (N, D)
            lp = F.log_softmax(self.heads[l](h[:, None, :] + prefix[None]), -1)   # (B, N, V)
            total = total + lp.gather(-1, codes[:, l][None, :, None].expand(B, N, 1)).squeeze(-1)
        return total


class GenerativeRec(Recommender):
    def __init__(self, dim=32, layers=1, max_len=30, levels=3, codebook=16, epochs=40,
                 lr=2e-3, batch_size=128, dropout=0.2, weight_decay=1e-6, seed=0):
        self.dim, self.layers, self.max_len, self.levels, self.codebook = dim, layers, max_len, levels, codebook
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.dropout, self.wd, self.seed = dropout, weight_decay, seed
        self.name = f"Generative(d={dim},L={layers})"

    def set_context(self, df: pd.DataFrame):
        """Which interactions count as 'the story so far' (the Evaluator sets this)."""
        self.seqs = build_sequences(df, self.n_users)

    def semantic_id(self, item_id: int):
        return tuple(int(v) for v in self.codes[item_id])

    def fit(self, data: Dataset):
        set_seed(self.seed)
        self.n_users, self.n_items = data.n_users, data.n_items
        self.set_context(data.train)

        # item addresses come from DESCRIPTIONS only (known for every item, including new ones)
        text = build_item_features(data, text_dim=64)[0]
        self.codes = semantic_ids(text, self.levels, self.codebook, self.seed)       # (N, n_lv)
        self.sizes = (self.codes.max(0) + 1).tolist()
        self.t_codes = torch.from_numpy(self.codes)
        pad = np.zeros((1, self.codes.shape[1]), dtype=np.int64)
        self.codes_in = torch.from_numpy(np.vstack([pad, self.codes + 1]))

        wins = [w for s in self.seqs for w in make_windows(s, self.max_len)]
        tokens = left_pad(wins, self.max_len + 1)
        rng = np.random.default_rng(self.seed)
        self.net = _GenNet(self.sizes, self.dim, self.max_len, self.layers, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(tokens))
            for s in range(0, len(order), self.batch_size):
                tok = torch.from_numpy(tokens[order[s:s + self.batch_size]])
                x, y = tok[:, :-1], tok[:, 1:]
                valid = (x != 0) & (y != 0)                         # real input AND real next item
                h = self.net.hidden(x, self.codes_in)[valid]        # (M, D)
                tgt = self.t_codes[y[valid] - 1]                    # (M, n_lv) true address of the next item
                loss = 0.0
                for l in range(len(self.sizes)):                    # teacher forcing: earlier tokens given
                    lg = self.net.code_logits(h, l, [tgt[:, j] for j in range(l)])
                    loss = loss + F.cross_entropy(lg, tgt[:, l])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, user_ids):
        """Log-probability of writing each item's address (higher = better)."""
        x = torch.from_numpy(left_pad([self.seqs[int(u)] for u in user_ids], self.max_len))
        out = []
        for s in range(0, len(x), 128):
            h = self.net.hidden(x[s:s + 128], self.codes_in)[:, -1]
            out.append(self.net.item_logprobs(h, self.t_codes).numpy())
        return np.concatenate(out).astype(np.float32)

    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Recent liked items whose address shares the longest prefix with this item.
        -> [(item_id, number_of_shared_leading_tokens)]  (ties keep the most recent first)"""
        recent = self.seqs[user_id][-self.max_len:][::-1]
        if len(recent) == 0:
            return []
        target = self.codes[item_id]

        def shared(i):
            eq = self.codes[i] == target
            return len(eq) if eq.all() else int(np.argmin(eq))

        pairs = sorted(((int(i), shared(i)) for i in recent), key=lambda p: -p[1])
        return pairs[:n]
