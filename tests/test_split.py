import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import preprocess  # noqa: E402
from recsys.data.split import temporal_split  # noqa: E402


def test_no_time_leakage():
    inter, items, u, i = preprocess.build(use_synthetic=True)
    train, val, test = temporal_split(inter)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_kcore_holds():
    inter, *_ = preprocess.build(use_synthetic=True)
    assert inter.groupby("user_id").size().min() >= 5
    assert inter.groupby("item_id").size().min() >= 5
