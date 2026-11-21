from pathlib import Path

import pytest

from recsys.serving.service import BUNDLE

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="run build_artifacts.py first")


def test_app_starts_without_errors():
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(path), default_timeout=180).run()
    assert not at.exception
