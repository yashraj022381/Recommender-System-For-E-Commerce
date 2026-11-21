"""Sequential recommenders (PyTorch): GRU4Rec-style and SASRec-style next-item models.

Each user's liked items, in time order, form one sequence. We treat a whole history as one
'session'; with real session data you would split sequences at long time gaps.
"""
import numpy
import torch

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from recsys.models.neural import set_seed


def build_sequences(df: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """Per user: liked item ids in time order (empty array if none)."""
    pos = df[df["label"] == 1].sort_values(["timestamp", "item_id"], kind="stable")
    seqs = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
    for u, items in pos.groupby("user_id")["item_id"]:
        seqs[int(u)] = items.to_numpy(dtype=np.int64)
    return seqs



def make_windows(seq: np.ndarray, L: int) -> list[np.ndarray]:
    """Windows of up to L+1 items, newest first, overlapping by one item,
    so every (item -> next item) transition is a training target exactly once."""
    out, end = [], len(seq)
    while end >= 2:
        start = max(0, end - (L + 1))
        out.append(seq[start:end])
        if start == 0:
            break
        end = start + 1
    return out


def left_pad(arrs, length: int) -> np.ndarray:
    """Item ids + 1 (0 = padding), padded on the LEFT so the newest item is always last."""
    out = np.zeros((len(arrs), length), dtype=np.int64)
    for r, a in enumerate(arrs):
        a = np.asarray(a)[-length:]
        if len(a):
            out[r, length - len(a):] = a + 1
    return out


class _Block(nn.Module):
    """Pre-norm Transformer block with an explicit attention mask."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)


    def forward(self, x, allow):                      # allow: (B, 1, L, L) bool, True = may attend
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))



class _SeqNet(nn.Module):
    def __init__(self, n_items, dim, max_len, kind, layers, dropout):
        super().__init__()
        self.kind = kind
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std = 0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.drop = nn.Dropout(dropout)
        if kind == "gru":
            self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        else:
            self.pos_emb = nn.Embedding(max_len, dim)
            self.blocks = nn.ModuleList([_Block(dim, 2, dropout) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)


    def hidden(self, x):                              # x: (B, L) of item ids + 1
        e = self.drop(self.item_emb(x))
        if self.kind == "gru":
            return self.gru(e)[0]
        B, L = x.shape
        e = e + self.pos_emb(torch.arange(L))[None]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]       # look back only
        allow = causal & ~(x == 0)[:, None, :]                              # ...and not at padding
        allow = (allow | torch.eye(L, dtype=torch.bool)[None])[:, None]     # always see yourself (no NaN rows)
        for blk in self.blocks:
            e = blk(e, allow)
        return self.norm(e)


    def item_table(self):
        return self.item_emb.weight[1:]               # (n_items, dim), tied with the output layer


class _SeqRec(Recommender):
    kind, label = "gru", "GRU4REC"

    def __init__(self, dim=32, layers=1, max_len=30, epochs=20, lr=2e-3,  batch_size=128,
                 dropout=0.2, weight_decay=1e-6, seed=0):
        self.dim, self.layers, self.max_len, self.epochs = dim, layers, max_len, epochs
        self.lr, self.batch_size, self.dropout, self.wd, self.seed = lr, batch_size, dropout, weight_decay, seed
        self.name = f"{self.label}(d={dim},L={layers})"

    def set_context(self, df: pd.DataFrame):
        """Which interactions count as 'the story so far' at scoring time.
        The Evaluator sets this to train (for val) or train+val (for test)."""
        self.seqs = build_sequences(df, self.n_users)


    def fit(self, data: Dataset):
        set_seed(self.seed)
        self.n_users, self.n_items = data.n_users, data.n_items
        self.set_context(data.train)
        wins = [w for s in self.seqs for w in make_windows(s, self.max_len)]
        tokens = left_pad(wins, self.max_len + 1)
        rng = np.random.default_rng(self.seed)
        self.net = _SeqNet(self.n_items, self.dim, self.max_len, self.kind, self.layers, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)


        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(tokens))
            for s in range(0, len(order), self.batch_size):
                tok = torch.from_numpy(tokens[order[s:s + self.batch_size]])
                x, y = tok[:, :-1], tok[:, 1:]
                valid = (x != 0) & (y != 0)                       # real input AND real next item
                logits = self.net.hidden(x) @ self.net.item_table().T   # (B, L, n_items)
                loss = F.cross_entropy(logits[valid], y[valid] - 1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, user_ids):
        x = torch.from_numpy(left_pad([self.seqs[int(u)] for u in user_ids], self.max_len))
        table = self.net.item_table()
        out = [(self.net.hidden(x[s:s + 512])[:, -1] @ table.T).numpy() for s in range(0, len(x), 512)]
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Recent liked items whose codes are closest to the recommended item. -> [(item_id, cosine)]"""
        recent = self.seqs[user_id][-self.max_len:]
        if len(recent) == 0:
            return []
        E = self.net.item_table()
        sims = F.cosine_similarity(E[torch.from_numpy(recent)], E[item_id][None], dim=1).numpy()
        order = np.argsort(-sims)[:n]
        return [(int(recent[j]), float(sims[j])) for j in order]


class GRU4RecRec(_SeqRec):
    kind, label = "gru", "GRU4Rec"

class SASRecRec(_SeqRec):
    kind, label = "sasrec", "SASRec"

    
    
        
 
    
    
