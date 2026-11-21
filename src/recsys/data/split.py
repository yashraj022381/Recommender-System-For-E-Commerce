"""Temporal (time-based) split + cold-start flags."""
import pandas as pd

from recsys.config import DATA


def temporal_split(inter: pd.DataFrame):
    """Oldest 80% -> train, next 10% -> val, newest 10% -> test (global time cutoffs)."""
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    t_train = inter["timestamp"].quantile(DATA.train_frac)
    t_val = inter["timestamp"].quantile(DATA.train_frac + DATA.val_frac)

    train = inter[inter["timestamp"] <= t_train]
    val = inter[(inter["timestamp"] > t_train) & (inter["timestamp"] <= t_val)]
    test = inter[inter["timestamp"] > t_val]

    # mark cold-start rows (user/item never seen in training)
    seen_u, seen_i = set(train["user_id"]), set(train["item_id"])
    for df in (val, test):
        df = df  # (views; we copy below)
    val = val.assign(
        cold_user=~val["user_id"].isin(seen_u),
        cold_item=~val["item_id"].isin(seen_i),
    )
    test = test.assign(
        cold_user=~test["user_id"].isin(seen_u),
        cold_item=~test["item_id"].isin(seen_i),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def describe(train, val, test, n_users, n_items) -> str:
    n = len(train) + len(val) + len(test)
    density = len(train) / (n_users * n_items)
    lines = [
        f"users={n_users:,}  items={n_items:,}  interactions={n:,}",
        f"train/val/test = {len(train):,} / {len(val):,} / {len(test):,}",
        f"train matrix density = {density:.5%}  (sparsity = {1 - density:.3%})",
        f"test rows with cold user = {test['cold_user'].mean():.1%}",
        f"test rows with cold item = {test['cold_item'].mean():.1%}",
    ]
    return "\n".join(lines)
