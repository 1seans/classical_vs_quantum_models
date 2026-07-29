from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception


def test_pages_switch_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value("Greeks Explorer").run()
    assert not at.exception


def test_data_loading_pages_render():
    # Exercises the data/*.npz artifacts + viz.surface/smile/convergence wiring,
    # the pages most likely to break on an artifact shape change.
    for page in ("Volatility Surfaces", "Quantum: QAE vs MC"):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised"
