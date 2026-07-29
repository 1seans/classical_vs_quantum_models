from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from models import black_scholes as bs


@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float


def simulate(S, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    logS = np.full(n_sims, np.log(S))
    v = np.full(n_sims, params.v0)
    sqrt_dt = np.sqrt(dt)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_sims)
        z3 = rng.standard_normal(n_sims)
        z2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * z3
        v_pos = np.maximum(v, 0.0)  # full truncation
        logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1
        v += params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * sqrt_dt * z2
    return np.exp(logS)


def price(S, K, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None):
    ST = simulate(S, T, r, params, q, n_sims, n_steps, seed)
    disc_payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    est = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(n_sims)
    return {"price": float(est), "stderr": float(stderr),
            "ci95": (float(est - 1.96 * stderr), float(est + 1.96 * stderr))}


def implied_vol(option_price, S, K, T, r, q=0.0):
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if option_price <= intrinsic + 1e-12:
        return 0.0
    f = lambda s: bs.call_price(S, K, T, r, s, q) - option_price
    try:
        return float(brentq(f, 1e-4, 5.0, maxiter=200))
    except ValueError:
        return float("nan")


def smile(S, T, r, params, strikes, q=0.0, **sim_kwargs):
    ivs = []
    for K in strikes:
        px = price(S, K, T, r, params, q, **sim_kwargs)["price"]
        ivs.append(implied_vol(px, S, K, T, r, q))
    return {"strikes": list(strikes), "implied_vols": ivs}
