"""Run:  python scripts/prepare_data.py            (real Amazon data)
         python scripts/prepare_data.py --synthetic (fast fake data)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.config import PATHS  # noqa: E402
from recsys.data import preprocess  # noqa: E402
from recsys.data.split import describe, temporal_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    PATHS.make()
    if not args.synthetic:
        from recsys.data.download import download_all
        download_all()

    inter, items, user_ids, item_ids = preprocess.build(use_synthetic=args.synthetic)
    train, val, test = temporal_split(inter)

    out = PATHS.processed
    train.to_parquet(out / "train.parquet")
    val.to_parquet(out / "val.parquet")
    test.to_parquet(out / "test.parquet")
    items.to_parquet(out / "items.parquet")
    (out / "meta.json").write_text(json.dumps({
        "n_users": len(user_ids), "n_items": len(item_ids),
        "synthetic": args.synthetic,
    }))
    (out / "id_maps.json").write_text(json.dumps({"user": user_ids, "item": item_ids}))

    print("\n=== DATA SUMMARY ===")
    print(describe(train, val, test, len(user_ids), len(item_ids)))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
