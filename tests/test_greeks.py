import pytest
from analytics import greeks


def test_analytic_matches_finite_difference():
    a = greeks.analytic(100, 100, 1.0, 0.05, 0.2, kind="call")
    fd = greeks.finite_difference(100, 100, 1.0, 0.05, 0.2, kind="call")
    for key in ("delta", "gamma", "vega", "theta", "rho"):
        assert a[key] == pytest.approx(fd[key], rel=1e-2, abs=1e-3)
