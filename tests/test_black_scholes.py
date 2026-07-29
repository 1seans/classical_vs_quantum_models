import numpy as np
from models import black_scholes as bs


def test_known_call_value():
    # Textbook: S=100,K=100,T=1,r=0.05,sigma=0.2,q=0 -> ~10.4506
    assert bs.call_price(100, 100, 1.0, 0.05, 0.2) == pytest_approx(10.4506, 1e-3)


def test_put_call_parity():
    S, K, T, r, sigma, q = 100, 110, 0.75, 0.03, 0.25, 0.01
    c = bs.call_price(S, K, T, r, sigma, q)
    p = bs.put_price(S, K, T, r, sigma, q)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest_approx(rhs, 1e-9)


def test_call_delta_bounds():
    d = bs.delta(100, 100, 1.0, 0.05, 0.2, kind="call")
    assert 0.0 < d < 1.0


def pytest_approx(v, tol):
    import pytest
    return pytest.approx(v, abs=tol) if tol >= 1e-3 else pytest.approx(v, rel=tol)
