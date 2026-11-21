import pytest

from recsys.serving.service import BUNDLE, RecService

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="run scripts/build_artifacts.py first")


@pytest.fixture(scope="module")
def svc():
    return RecService()


def test_every_model_returns_k_unowned_items(svc):
    u = svc.demo_users[0]
    owned = set(svc.history[u].indices.tolist())
    for name in svc.models:
        cards, ms = svc.recommend(name, u, 10)
        ids = [c["item_id"] for c in cards]
        assert len(ids) == 10 and len(set(ids)) == 10, name
        assert not (set(ids) & owned), name
        assert ms < 500, name


def test_similar_items_exclude_the_item_itself(svc):
    for method in svc.SIMILAR_METHODS:
        ids = [c["item_id"] for c in svc.similar_items(5, method, 8)]
        assert 5 not in ids and len(ids) == 8


def test_new_shopper_picks_drive_recommendations(svc):
    ids = [c["item_id"] for c in svc.from_items([3, 4], k=10)]
    assert len(ids) == 10 and not ({3, 4} & set(ids))


def test_live_mode_does_not_change_the_stored_history(svc):
    u = svc.demo_users[0]
    before = svc.raw["SASRec"].seqs[u].copy()
    cards, mode, ms = svc.live_recommend("SASRec", u, [1, 2], 10)
    assert (svc.raw["SASRec"].seqs[u] == before).all()
    assert not ({1, 2} & {c["item_id"] for c in cards})


def test_explanations_are_never_empty(svc):
    u = svc.demo_users[0]
    for name in svc.models:
        cards, _ = svc.recommend(name, u, 3)
        assert svc.explain(name, u, cards[0]["item_id"])["lines"], name


def test_cold_users_get_popularity(svc):
    if not svc.cold_users:
        pytest.skip("no cold users in this split")
    u = svc.cold_users[0]
    a = [c["item_id"] for c in svc.recommend("UserKNN", u, 10)[0]]
    b = [c["item_id"] for c in svc.recommend("Popularity", u, 10)[0]]
    assert a == b
