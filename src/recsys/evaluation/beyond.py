"""Beyond-accuracy metrics for the top-k lists a model shows."""
import numpy as np
import pandas as pd



def beyond_accuracy(model, ev, train_pop, item_sim, cats, k=10) -> pd.Series:
    """coverage      share of the catalog that appears in ANY user's top-k
       novelty       mean -log2(popularity share) of shown items (higher = less obvious items)
       intra_sim     mean pairwise similarity inside a list (lower = more varied)
       distinct_cats mean number of different categories in a list
       gini          concentration of exposure across items (0 = even, 1 = one item gets everything)"""
    if hasattr(model, "set_context"):
        model.set_context(ev.context)
    p = (train_pop + 1.0) / (train_pop + 1.0).sum()
    info = -np.log2(p)
    cats = np.asarray(cats)
    shown = np.zeros(len(train_pop))
    nov, sim, dc = [], [], []
    for s in range(0, len(ev.users), 1024):
        u = ev.users[s:s + 1024]
        sc = np.array(model.score(u), dtype=np.float32, copy=True)
        h = ev.history[u].tocoo()
        sc[h.row, h.col] = -np.inf
        top = ev._topk(sc)[:, :k]
        np.add.at(shown, top.ravel(), 1)
        for t in top:
            S = item_sim[np.ix_(t, t)]
            sim.append(float((S.sum() - np.trace(S)) / (k * (k - 1))))
            nov.append(float(info[t].mean()))
            dc.append(len(set(cats[t].tolist())))
        srt = np.sort(shown)
        n = len(srt)
        gini = float((2 * np.arange(1, n + 1) - n - 1) @ srt / (n * srt.sum() + 1e-9))
        return pd.Series({"coverage": float((shown > 0).mean()), "novelty": np.mean(nov),
                      "intra_sim": np.mean(sim), "distinct_cats": np.mean(dc), "gini": gini})                  
   
