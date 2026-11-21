"""Paired comparison of two models on the SAME users."""
import numpy as np
import pandas as pd


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0):
    """a, b: per-user metric values (index = user_id). Returns (mean_diff, ci_low, ci_high, p_two_sided).
    Resamples users; p is the share of resamples where the sign of the difference flips."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0). mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def run_seeds(make, data, ev, seeds=(0, 1, 2), metric="ndcg@10", wrap=None):
    """Fit `make(seed)` for each seed. Returns (per-user DataFrame averaged over seeds, list of scores)."""
    frames, scores = [], []
    for s in seeds:
        m = make(s).fit(data)
        m = wrap(m) if wrap else m
        df = ev.evaluate(m)
        frames.append(df[metric])
        scores.append(float(df[metric].mean()))
    return pd.concat(frames, axis=1).mean(1), scores
