"""Smoke test for the Streamlit app.

Runs only where a trained bundle exists (it is built locally by
scripts/run_pipeline.py and not tracked in git), so CI skips it.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "hdblens_bundle.joblib"


@pytest.mark.skipif(not BUNDLE.exists(), reason="no trained bundle; run scripts/run_pipeline.py")
def test_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception

    # Changing the town must rerun cleanly and move the estimate.
    hero = next(m for m in at.markdown if 'class="hero-card"' in m.value)
    at.selectbox[0].select("QUEENSTOWN").run()
    assert not at.exception
    new_hero = next(m for m in at.markdown if 'class="hero-card"' in m.value)
    assert new_hero.value != hero.value
