import json
from types import SimpleNamespace as NS

import pytest

from recsys.llm.assistant import Assistant, ShopTools
from recsys.llm.explain import explain_in_words, grounded
from recsys.serving.service import BUNDLE


def card(i, price=None, oos=False):
    return {"rank": i + 1, "item_id": i, "title": f"Item {i}", "category": "c",
            "price": 10.0 * (i + 1) if price is None else price, "score": 1.0,
            "new": False, "liked_by": 3, "oos": oos}


class StubSvc:
    n_items = 5
    SIMILAR_METHODS = ("Co-purchase", "Description (TF-IDF)", "Two-Tower codes")

    def search(self, q, k=10):
        return [card(i) for i in range(min(k, 5))]

    def similar_items(self, i, method, k=10):
        return [card(j) for j in range(min(k, 5)) if j != i]

    def recommend(self, model, user, k=10):
        return [card(i) for i in range(min(k, 5))], 0.0

    def from_items(self, ids, k=10, exclude=()):
        return [card(i) for i in range(min(k, 5)) if i not in ids]

    def _cards(self, items, scores=None):
        return [card(int(i)) for i in items]


class FakeClient:
    """Mimics client.chat.completions.create(...)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = NS(completions=self)

    def create(self, **kw):
        return self.replies.pop(0)


class BrokenClient:
    def __init__(self):
        self.chat = NS(completions=self)

    def create(self, **kw):
        raise RuntimeError("boom")


def tool_use(name, args):
    call = NS(id="t1", function=NS(name=name, arguments=json.dumps(args)))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])


def text(s):
    return NS(choices=[NS(message=NS(content=s, tool_calls=None))])


# ---------------- Option A ----------------
def test_check_accepts_grounded_text():
    assert grounded('This matches "Rpg Game 1" because 73% of your likes are rpg.',
                    "73% of your liked items are in 'rpg'.\nRpg Game 1")


def test_check_rejects_invented_number():
    assert not grounded("Only $5 today!", "Price $57.94 vs your usual $41.20.")


def test_check_rejects_invented_title():
    assert not grounded('You will love "Zelda 9".', "Liked by 21 shoppers so far.\nRpg Game 1")


def test_explain_uses_valid_model_text():
    lines = ["73% of your liked items are in 'rpg'."]
    client = FakeClient([text('"Rpg Game 1" fits: 73% of your likes are rpg.')])
    out, source = explain_in_words(client, "Rpg Game 1", lines)
    assert source == "llm" and "73%" in out


def test_explain_falls_back_on_invented_number():
    lines = ["73% of your liked items are in 'rpg'."]
    out, source = explain_in_words(FakeClient([text("A bargain at $5!")]), "Rpg Game 1", lines)
    assert source == "template (failed check)" and "73%" in out


def test_explain_without_client_uses_template():
    out, source = explain_in_words(None, "x", ["Liked by 3 shoppers so far."])
    assert source == "template" and "Liked by 3" in out


# ---------------- Option B ----------------
def test_price_filter():
    r = ShopTools(StubSvc()).call("search_products", {"query": "x", "k": 10, "max_price": 25})
    assert [p["id"] for p in r["products"]] == [0, 1]


def test_bad_item_id():
    assert "error" in ShopTools(StubSvc()).call("get_product", {"item_id": 99})


def test_assistant_happy_path():
    client = FakeClient([tool_use("search_products", {"query": "x", "k": 3}), text("Try Item 0 [id:0].")])
    out = Assistant(ShopTools(StubSvc()), client).ask("hi")
    assert "(#0)" in out["text"] and not out["flagged"] and out["used_llm"] and len(out["cards"]) == 3


def test_assistant_blocks_unknown_ids():
    client = FakeClient([tool_use("search_products", {"query": "x", "k": 3}), text("Buy [id:99] now!")])
    out = Assistant(ShopTools(StubSvc()), client).ask("hi")
    assert out["flagged"] and "[id:99]" not in out["text"] and out["cards"]


def test_assistant_without_client():
    out = Assistant(ShopTools(StubSvc()), None).ask("games")
    assert not out["used_llm"] and out["cards"]


def test_assistant_survives_api_error():
    out = Assistant(ShopTools(StubSvc()), BrokenClient()).ask("games")
    assert "unavailable" in out["text"] and out["cards"] and not out["used_llm"]


@pytest.mark.skipif(not BUNDLE.exists(), reason="run build_artifacts.py first")
def test_real_service_search():
    from recsys.serving.service import RecService
    cards = RecService().search("fantasy dragon quest", 5)
    assert 0 < len(cards) <= 5
