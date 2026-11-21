"""Synthetic e-commerce data: realistic sparsity, taste groups, and cold-start."""
import numpy as np
import pandas as pd

CATEGORIES = {
    "action":  ["fast", "shooter", "combat", "explosive", "mission", "weapon"],
    "puzzle":  ["brain", "logic", "relaxing", "match", "solve", "tiles"],
    "sports":  ["football", "racing", "league", "season", "team", "stadium"],
    "rpg":     ["quest", "fantasy", "dragon", "hero", "magic", "adventure"],
    "kids":    ["colorful", "friendly", "learn", "fun", "family", "animals"],
}


def make_synthetic(n_users=3000, n_items=800, seed=42):
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    K = len(cats)

    # ---- items ----
    item_cat = rng.integers(0, K, n_items)
    rank = rng.permutation(n_items)
    item_pop = 1.0 / (rank + 20) ** 0.7          # mild long-tail (not extreme)
    item_price = np.round(rng.lognormal(3.2, 0.5, n_items), 2)
    # ~8% of items launch late in the year -> cold-start items
    item_release = np.where(rng.random(n_items) < 0.08,
                            rng.uniform(0.90, 0.97, n_items), 0.0)

    items = pd.DataFrame({
        "item_raw": [f"I{i:05d}" for i in range(n_items)],
        "title": [f"{cats[c].title()} Game {i}" for i, c in enumerate(item_cat)],
        "text": [
            f"{cats[c].title()} Game {i}. " + " ".join(rng.choice(CATEGORIES[cats[c]], 8))
            for i, c in enumerate(item_cat)
        ],
        "category": [cats[c] for c in item_cat],
        "price": item_price,
    })

    # ---- users ----
    taste = rng.dirichlet(np.ones(K) * 0.4, n_users)         # each user prefers 1-2 categories
    activity = np.clip(rng.lognormal(2.9, 0.6, n_users), 8, 80).astype(int)
    # ~6% of users join late -> cold-start users
    join = np.where(rng.random(n_users) < 0.06,
                    rng.uniform(0.90, 0.97, n_users), rng.uniform(0.0, 0.80, n_users))

    t0 = pd.Timestamp("2022-01-01").value // 10**6
    span = 365 * 24 * 3600 * 1000

    rows = []
    for u in range(n_users):
        w = item_pop * (0.05 + taste[u][item_cat])
        w = w / w.sum()
        n = min(activity[u], n_items)
        chosen = rng.choice(n_items, size=n, replace=False, p=w)   # distinct items
        for i in chosen:
            start = max(join[u], item_release[i])
            t = rng.uniform(start, 1.0)
            liked = taste[u][item_cat[i]] > 0.25
            rating = float(np.clip(round(rng.normal(4.3 if liked else 3.2, 0.8)), 1, 5))
            rows.append((f"U{u:05d}", f"I{i:05d}", rating, int(t0 + t * span)))

    inter = pd.DataFrame(rows, columns=["user_raw", "item_raw", "rating", "timestamp"])
    return inter, items
