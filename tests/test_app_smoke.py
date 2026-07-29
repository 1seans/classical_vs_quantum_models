from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception


def test_pages_switch_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value("Greeks Explorer").run()
    assert not at.exception
