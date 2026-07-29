import numpy as np
from models import black_scholes as bs


def price(S, K, T, r, sigma, q=0.0, n_sims=100_000, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc_payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    est = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(n_sims)
    return {"price": float(est), "stderr": float(stderr),
            "ci95": (float(est - 1.96 * stderr), float(est + 1.96 * stderr))}


def convergence(S, K, T, r, sigma, q=0.0, sample_sizes=(1000, 5000, 20000, 100000, 400000), seed=0):
    exact = bs.call_price(S, K, T, r, sigma, q)
    ns, errs = [], []
    for n in sample_sizes:
        est = price(S, K, T, r, sigma, q, n_sims=int(n), seed=seed)["price"]
        ns.append(int(n))
        errs.append(abs(est - exact))
    return {"n": ns, "abs_error": errs}
