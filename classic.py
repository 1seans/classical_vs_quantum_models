# classic.py

import numpy as np
import plotly.express as px
from scipy.stats import norm
from utils_plotly import create_3d_surface

def black_scholes_call(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def generate_classical_surface(S0, T, r, sigma_input, q):
    vol_range = np.linspace(0.1, 0.6, 25)
    T_range = np.linspace(0.1, 1.0, 25)
    X, Y = np.meshgrid(vol_range, T_range)

    Z = np.array([[black_scholes_call(S0, S0, t, r, s, q) for s in vol_range] for t in T_range])
    return create_3d_surface(vol_range, T_range, Z, title="Classical GBM Surface",
                             x_label="Volatility (σ)", y_label="Time to Maturity (T)", z_label="Call Price")

def monte_carlo_gbm(S0, mu, sigma, T, n_steps, n_simulations):
    dt = T / n_steps
    prices = np.zeros((n_simulations, n_steps))
    prices[:, 0] = S0
    for t in range(1, n_steps):
        rand = np.random.normal(0, 1, n_simulations)
        prices[:, t] = prices[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
    return prices

def classical_simulation_results(S0, K, T, r, sigma, n_simulations):
    n_steps = 252
    mu = 0.05
    time_grid = np.linspace(0, T, n_steps)
    prices = monte_carlo_gbm(S0, mu, sigma, T, n_steps, n_simulations)
    final_prices = prices[:, -1]
    avg_path = np.mean(prices, axis=0)
    payoff = np.maximum(final_prices - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    pnl = final_prices - S0

    import plotly.graph_objects as go
    charts = []

    # Sample paths
    fig1 = go.Figure()
    for i in range(min(10, n_simulations)):
        fig1.add_trace(go.Scatter(y=prices[i], mode='lines', line=dict(width=1)))
    fig1.update_layout(title="Simulated Price Paths", xaxis_title="Time", yaxis_title="Price")
    charts.append(("Simulated Paths", fig1))

    # Histogram
    fig2 = px.histogram(final_prices, nbins=30, title="Histogram of Final Prices")
    charts.append(("Final Price Distribution", fig2))

    # Average Path
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=avg_path, mode='lines', name='Average Path'))
    fig3.update_layout(title="Average Path", xaxis_title="Time", yaxis_title="Price")
    charts.append(("Average Path", fig3))

    error_pct = ((np.mean(final_prices) - S0) / S0) * 100

    # Summary
    summary = (
        f"Expected Final Price: ${np.mean(final_prices):.2f}\n"
        f"Std Dev: ${np.std(final_prices):.2f}\n"
        f"Option Price: ${option_price:.2f}\n"
        f"P&L Volatility: ${np.std(pnl):.2f}\n"
        f"% Error from Initial Price: {error_pct:.2f}%"
    )

    return {"charts": charts, "summary": summary}
