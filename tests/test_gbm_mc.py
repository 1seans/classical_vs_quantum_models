import pytest
from models import gbm_mc
from models import black_scholes as bs


def test_mc_converges_to_black_scholes_within_ci():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    exact = bs.call_price(S, K, T, r, sigma)
    out = gbm_mc.price(S, K, T, r, sigma, n_sims=400_000, seed=42)
    lo, hi = out["ci95"]
    assert lo <= exact <= hi
    assert out["stderr"] > 0


def test_convergence_error_shrinks():
    out = gbm_mc.convergence(100, 100, 1.0, 0.05, 0.2, sample_sizes=(1000, 200_000), seed=1)
    assert out["abs_error"][-1] < out["abs_error"][0]
