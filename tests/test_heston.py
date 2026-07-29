import pytest
from models import heston
from models import black_scholes as bs


def test_heston_collapses_to_black_scholes_when_xi_zero():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    params = heston.HestonParams(kappa=1.0, theta=sigma**2, xi=0.0, rho=0.0, v0=sigma**2)
    out = heston.price(S, K, T, r, params, n_sims=200_000, n_steps=200, seed=7)
    exact = bs.call_price(S, K, T, r, sigma)
    lo, hi = out["ci95"]
    assert lo <= exact <= hi


def test_implied_vol_roundtrip():
    px = bs.call_price(100, 100, 1.0, 0.05, 0.3)
    iv = heston.implied_vol(px, 100, 100, 1.0, 0.05)
    assert iv == pytest.approx(0.3, abs=1e-4)


def test_smile_has_curvature_under_negative_rho():
    params = heston.HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    out = heston.smile(100, 1.0, 0.05, params, strikes=[80, 100, 120], n_sims=120_000, n_steps=100, seed=3)
    ivs = out["implied_vols"]
    assert ivs[0] != pytest.approx(ivs[2], abs=1e-3)  # skew: wings differ
