import pytest
from quantum import qae_pricing
from models import black_scholes as bs


def test_qae_matches_black_scholes_within_tolerance():
    S, K, T, r, sigma = 2.0, 1.9, 40 / 365, 0.05, 0.4
    exact = bs.call_price(S, K, T, r, sigma)
    out = qae_pricing.price(S, K, T, r, sigma, num_qubits=3, epsilon=0.005)
    # QAE on 3 qubits discretizes the distribution; tolerance reflects that.
    assert out["price"] == pytest.approx(exact, abs=0.05)
    assert out["circuit"] is not None


def test_convergence_returns_both_series():
    out = qae_pricing.convergence_vs_mc(2.0, 1.9, 40 / 365, 0.05, 0.4,
                                        num_qubits=3, mc_sample_sizes=(500, 5000))
    assert len(out["mc"]["n"]) == 2
    assert len(out["qae"]["queries"]) >= 1
