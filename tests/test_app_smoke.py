from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_dashboard_renders_without_streamlit_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    result = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert len(result.exception) == 0, [exception.message for exception in result.exception]
