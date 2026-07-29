from models import black_scholes as bs


def analytic(S, K, T, r, sigma, q=0.0, kind="call"):
    return {
        "delta": bs.delta(S, K, T, r, sigma, q, kind),
        "gamma": bs.gamma(S, K, T, r, sigma, q),
        "vega": bs.vega(S, K, T, r, sigma, q),
        "theta": bs.theta(S, K, T, r, sigma, q, kind),
        "rho": bs.rho(S, K, T, r, sigma, q, kind),
    }


def _px(S, K, T, r, sigma, q, kind):
    return bs.call_price(S, K, T, r, sigma, q) if kind == "call" else bs.put_price(S, K, T, r, sigma, q)


def finite_difference(S, K, T, r, sigma, q=0.0, kind="call"):
    hS, hV, hR, hT = S * 1e-4, 1e-4, 1e-5, 1e-5
    p = lambda **kw: _px(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), kw.get("sigma", sigma), q, kind)
    delta = (p(S=S + hS) - p(S=S - hS)) / (2 * hS)
    gamma = (p(S=S + hS) - 2 * p() + p(S=S - hS)) / hS**2
    vega = (p(sigma=sigma + hV) - p(sigma=sigma - hV)) / (2 * hV)
    theta = -(p(T=T + hT) - p(T=T - hT)) / (2 * hT)
    rho = (p(r=r + hR) - p(r=r - hR)) / (2 * hR)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
