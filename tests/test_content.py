import numpy as np
import pandas as pd
import pytest

from recsys.data.loader import Dataset
from recsys.models.content import TfidfContentRec


def tiny():
    """Items 0,1 are 'dragon quest' items; 2,3 are 'puzzle logic' items.
    Item 1 has NO interactions (cold). User 0 liked item 0 only."""
    items = pd.DataFrame({
        "item_id": [0, 1, 2, 3],
        "title": ["Dragon Quest", "Dragon Quest Two", "Logic Puzzle", "Puzzle Tiles"],
        "text": ["dragon quest hero magic", "dragon quest hero castle",
                 "logic puzzle brain", "puzzle tiles match"],
    })
    train = pd.DataFrame({"user_id": [0, 1, 1], "item_id": [0, 2, 3], "label": [1, 1, 1]})
    empty = train.iloc[:0]
    return Dataset(train, empty, empty, items, n_users=2, n_items=4)


def test_cold_item_is_recommended_by_text():
    data = tiny()
    m = TfidfContentRec().fit(data)
    s = m.score(np.array([0]))[0]
    assert s[1] > s[2] and s[1] > s[3]          # cold dragon item beats puzzle items


def test_top_terms_explain():
    data = tiny()
    m = TfidfContentRec().fit(data)
    words = [w for w, _ in m.top_terms(0, 1)]
    assert any("dragon" in w for w in words)
