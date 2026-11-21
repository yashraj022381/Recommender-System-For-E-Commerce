"""Load -> clean -> k-core filter -> map IDs -> save parquet."""
import gzip
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys.config import DATA, PATHS


# ---------- loading ----------
def _read_jsonl_gz(path, max_rows=None):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(tqdm(f, desc=f"reading {path.name}")):
            if max_rows and n >= max_rows:
                break
            yield json.loads(line)


def load_reviews() -> pd.DataFrame:
    rows = [
        (r["user_id"], r["parent_asin"], float(r["rating"]), int(r["timestamp"]))
        for r in _read_jsonl_gz(PATHS.raw / "reviews.jsonl.gz", DATA.max_review_rows)
    ]
    return pd.DataFrame(rows, columns=["user_raw", "item_raw", "rating", "timestamp"])


def _join(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return "" if x is None else str(x)


def load_meta() -> pd.DataFrame:
    rows = []
    for r in _read_jsonl_gz(PATHS.raw / "meta.jsonl.gz"):
        cats = r.get("categories") or []
        price = r.get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = np.nan
        title = r.get("title") or ""
        text = " ".join([title, _join(r.get("features")), _join(r.get("description"))]).strip()
        rows.append((
            r["parent_asin"], title, text,
            cats[-1] if cats else (r.get("main_category") or "unknown"),
            price,
        ))
    return pd.DataFrame(rows, columns=["item_raw", "title", "text", "category", "price"])


# ---------- cleaning ----------
def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["user_raw", "item_raw", "timestamp"]).copy()
    # if a user rated an item twice, keep the first time (avoids leakage)
    df = df.sort_values("timestamp").drop_duplicates(["user_raw", "item_raw"], keep="first")
    return df


def k_core(df: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """Repeat until every user & item has enough interactions."""
    while True:
        before = len(df)
        uc = df["user_raw"].value_counts()
        df = df[df["user_raw"].isin(uc[uc >= min_u].index)]
        ic = df["item_raw"].value_counts()
        df = df[df["item_raw"].isin(ic[ic >= min_i].index)]
        if len(df) == before:
            return df.reset_index(drop=True)


# ---------- ID mapping ----------
def map_ids(inter: pd.DataFrame, items: pd.DataFrame):
    user_ids = {u: i for i, u in enumerate(sorted(inter["user_raw"].unique()))}
    item_ids = {x: i for i, x in enumerate(sorted(inter["item_raw"].unique()))}

    inter = inter.copy()
    inter["user_id"] = inter["user_raw"].map(user_ids).astype("int32")
    inter["item_id"] = inter["item_raw"].map(item_ids).astype("int32")

    items = items[items["item_raw"].isin(item_ids)].drop_duplicates("item_raw").copy()
    items["item_id"] = items["item_raw"].map(item_ids).astype("int32")
    # keep items that have interactions but no metadata (cold text) too
    missing = set(item_ids) - set(items["item_raw"])
    if missing:
        extra = pd.DataFrame({
            "item_raw": list(missing), "title": "", "text": "",
            "category": "unknown", "price": np.nan,
        })
        extra["item_id"] = extra["item_raw"].map(item_ids).astype("int32")
        items = pd.concat([items, extra], ignore_index=True)
    items = items.sort_values("item_id").reset_index(drop=True)
    items["price"] = items["price"].fillna(items["price"].median())
    return inter, items, user_ids, item_ids


def build(use_synthetic: bool = False):
    if use_synthetic:
        from recsys.data.synthetic import make_synthetic
        inter, items = make_synthetic()
    else:
        inter, items = load_reviews(), load_meta()
        if inter.empty:
            raise RuntimeError("No reviews were read. The file in data/raw is empty or damaged: "
                               "delete it and re-run prepare_data.py.")

    inter = clean_interactions(inter)
    inter = k_core(inter, DATA.min_user_interactions, DATA.min_item_interactions)
    inter, items, user_ids, item_ids = map_ids(inter, items)

    # implicit label: did the user "like" it?
    inter["label"] = (inter["rating"] >= DATA.positive_rating_threshold).astype("int8")
    return inter, items, user_ids, item_ids
