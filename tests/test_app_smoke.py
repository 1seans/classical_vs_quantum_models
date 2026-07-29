import subprocess
import sys

from streamlit.testing.v1 import AppTest


def test_app_startup_does_not_import_qiskit():
    # Regression guard: qiskit must stay a lazy import (live-QAE button only).
    # If it leaks into startup, a bad qiskit on the host takes the whole site down.
    code = ("import sys; from precompute import build_data; import app; "
            "assert not any(m.startswith('qiskit') for m in sys.modules), 'qiskit imported at startup'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


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
    for page in ("Volatility Surfaces", "Quantum: QAE vs MC", "The Verdict"):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised"
