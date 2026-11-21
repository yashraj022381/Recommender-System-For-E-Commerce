"""Beyond-accuracy view of the top-k lists a model actually shows."""
import numpy as np
import pandas as pd


def list_stats(model, ev, item_sim, cats, fresh, oos=None, k: int = 10) -> pd.Series:
    if hasattr(model, "set_context"):
        model.set_context(ev.context)
    cats, fresh = np.asarray(cats), np.asarray(fresh)
    distinct, fresh_share, sim, n_oos = [], [], [], []
    for s in range(0, len(ev.users), 1024):
        u = ev.users[s:s + 1024]
        sc = np.array(model.score(u), dtype=np.float32, copy=True)
        h = ev.history[u].tocoo()
        sc[h.row, h.col] = -np.inf
        for t in ev._topk(sc)[:, :k]:
            S = item_sim[np.ix_(t, t)]
            distinct.append(len(set(cats[t].tolist())))
            fresh_share.append(float(fresh[t].mean()))
            sim.append(float((S.sum() - np.trace(S)) / (k * (k - 1))))
            if oos is not None:
                n_oos.append(int(oos[t].sum()))
    out = {f"distinct_cats@{k}": np.mean(distinct),
           f"fresh_share@{k}": np.mean(fresh_share),
           f"intra_list_sim@{k}": np.mean(sim)}
    if oos is not None:
        out[f"oos_in_top{k}"] = np.mean(n_oos)
    return pd.Series(out)
