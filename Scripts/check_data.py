"""Run:  python check_data.py     Is the processed data usable? Read this BEFORE training anything."""
import json

import numpy as np
import pandas as pd

from recsys.config import PATHS
from recsys.data.loader import load_dataset


def main():
    d = load_dataset()
    meta = json.loads((PATHS.processed / "meta.json").read_text())
    print(f"synthetic: {meta.get('synthetic')}")
    print(f"users={d.n_users:,}  items={d.n_items:,}  train/val/test={len(d.train):,}/{len(d.val):,}/{len(d.test):,}")
    density = len(d.train) / (d.n_users * d.n_items)
    print(f"train density = {density:.5%}")
    per_user = d.train.groupby("user_id").size()
    print(f"interactions per user (train): median {per_user.median():.0f}, mean {per_user.mean():.1f}, max {per_user.max()}")
    print(f"test rows with cold user = {d.test['cold_user'].mean():.1%}, cold item = {d.test['cold_item'].mean():.1%}")
    print(f"positive rate (rating >= 4): {d.train['label'].mean():.1%}")

    for name, df in (("train", d.train), ("val", d.val), ("test", d.test)):
        t = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"  {name:<5} {t.min().date()} -> {t.max().date()}")

    txt = (d.items["title"].fillna("") + " " + d.items["text"].fillna("")).str.len()
    print(f"\nitems with >= 50 characters of text: {(txt >= 50).mean():.1%}   (median {txt.median():.0f} chars)")
    vc = d.items["category"].value_counts()
    print(f"distinct categories: {len(vc)}   top: {dict(vc.head(5))}")
    print(f"prices: median ${d.items['price'].median():.2f}")

    print("\nWARNINGS")
    warned = False
    if d.n_items > 8000:
        print(f"  - {d.n_items:,} items: the app's dense item-item tables will not fit. Raise RECSYS_MIN_ITEM."); warned = True
    if len(vc) < 4:
        print("  - almost no category information: the 'fresh x favorite' rule and the category cap "
              "become meaningless on this data."); warned = True
    if (txt >= 50).mean() < 0.6:
        print("  - most items have little text: content-based and Generative models will be weak."); warned = True
    if d.test["cold_user"].mean() > 0.3:
        print("  - over 30% of test rows come from users unseen in training: expect popularity to dominate."); warned = True
    if not warned:
        print("  none")


if __name__ == "__main__":
    main()
