from pathlib import Path

import pytest

from recsys.serving.service import BUNDLE

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="run build_artifacts.py first")
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=180).run()


def test_product_mode_renders():
    at = _app()
    at.radio[0].set_value("A product").run()
    assert not at.exception


def test_live_click_updates_session():
    at = _app()
    [b for b in at.button if b.label == "Click"][0].click().run()
    assert not at.exception
    assert len(at.session_state["clicks"]) == 1


def test_ab_stream_runs():
    at = _app()
    [b for b in at.button if "Start stream" in b.label][0].click().run()
    assert not at.exception
