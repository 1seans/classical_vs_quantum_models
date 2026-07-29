import numpy as np
from scipy.optimize import minimize
from models import heston


def calibrate(S, T, r, market, q=0.0, seed=0):
    strikes = [k for k, _ in market]
    target = np.array([iv for _, iv in market])

    def loss(x):
        kappa, theta, xi, rho, v0 = x
        params = heston.HestonParams(kappa, theta, xi, rho, v0)
        model = heston.smile(S, T, r, params, strikes, q=q,
                             n_sims=60_000, n_steps=60, seed=seed)
        ivs = np.array(model["implied_vols"])
        if np.any(np.isnan(ivs)):
            return 1e6
        return float(np.mean((ivs - target) ** 2))

    x0 = [2.0, float(np.mean(target) ** 2), 0.3, -0.5, float(np.mean(target) ** 2)]
    bounds = [(0.1, 10), (0.001, 0.5), (0.01, 2.0), (-0.95, 0.0), (0.001, 0.5)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40})
    if not res.success and res.fun > 1e-2:
        return None
    k, th, xi, rho, v0 = res.x
    return heston.HestonParams(k, th, xi, rho, v0)


def fetch_market_smile(ticker):
    import yfinance as yf
    from datetime import datetime
    tk = yf.Ticker(ticker)
    spot_hist = tk.history(period="5d")
    if spot_hist.empty:
        return None
    S = float(spot_hist["Close"].iloc[-1])
    expiries = tk.options
    if not expiries:
        return None
    expiry = expiries[min(3, len(expiries) - 1)]
    chain = tk.option_chain(expiry).calls
    chain = chain[(chain["impliedVolatility"] > 0.01) & (chain["strike"] > 0)]
    if chain.empty:
        return None
    T = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days, 1) / 365
    mkt = list(zip(chain["strike"].astype(float), chain["impliedVolatility"].astype(float)))
    mkt = [m for m in mkt if 0.5 * S < m[0] < 1.5 * S][:9]
    return (S, T, mkt) if mkt else None
