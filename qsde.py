import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer
from utils_plotly import create_3d_surface
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def quantum_probability_adjustment():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    backend = AerSimulator()
    qc.save_statevector()
    result = backend.run(transpile(qc, backend)).result()
    statevector = result.data(0)['statevector']
    probabilities = np.abs(statevector) ** 2
    return probabilities[0]


def hybrid_simulation(S0, mu, sigma, T, n_steps, n_simulations):
    dt = T / n_steps
    prices = np.zeros((n_simulations, n_steps))
    prices[:, 0] = S0
    for i in range(n_simulations):
        quantum_adjustment = quantum_probability_adjustment()
        bias = (quantum_adjustment - 0.42) * 0.1
        for t in range(1, n_steps):
            rand = np.random.normal(0, 1)
            drift = ((mu + bias) - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * rand
            prices[i, t] = prices[i, t - 1] * np.exp(drift + diffusion)
    return prices


@st.cache_data(show_spinner=True)
def generate_qsde_surface(S0, T, r, sigma_input, q):
    vol_range = np.linspace(0.1, 0.6, 32)
    T_range = np.linspace(0.1, 1.0, 32)
    cache_path = "data/qsde_surface.npy"

    # Try loading saved surface
    if os.path.exists(cache_path):
        Z = np.load(cache_path)
        return create_3d_surface(
            vol_range, T_range, Z,
            title="Quantum GBM Surface (cached)",
            x_label="Volatility (σ)", y_label="Time to Maturity (T)", z_label="Call Price"
        )

    # Otherwise compute it
    Z = np.zeros((len(T_range), len(vol_range)))
    for i, t in enumerate(T_range):
        for j, sigma in enumerate(vol_range):
            sim_prices = hybrid_simulation(
                S0=S0, mu=0.05, sigma=sigma,
                T=t, n_steps=252, n_simulations=100
            )
            final_prices = sim_prices[:, -1]
            payoff = np.maximum(final_prices - S0, 0)
            Z[i, j] = np.exp(-r * t) * np.mean(payoff)

    # Save for future runs
    np.save(cache_path, Z)

    return create_3d_surface(
        vol_range, T_range, Z,
        title="Quantum GBM Surface (computed)",
        x_label="Volatility (σ)", y_label="Time to Maturity (T)", z_label="Call Price"
    )


def qsde_simulation_results(S0, K, T, r, sigma, n_simulations):
    n_steps = 252
    mu = 0.05
    time_grid = np.linspace(0, T, n_steps)
    prices = hybrid_simulation(S0, mu, sigma, T, n_steps, n_simulations)
    final_prices = prices[:, -1]
    avg_path = np.mean(prices, axis=0)
    payoff = np.maximum(final_prices - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    pnl = final_prices - S0

    charts = []

    fig1 = go.Figure()
    for i in range(min(10, n_simulations)):
        fig1.add_trace(go.Scatter(y=prices[i], mode='lines', line=dict(width=1)))
    fig1.update_layout(title="Simulated Price Paths (QSDE)", xaxis_title="Time", yaxis_title="Price")
    charts.append(("Simulated Paths", fig1))

    fig2 = px.histogram(final_prices, nbins=30, title="Histogram of Final Prices (QSDE)")
    charts.append(("Final Price Distribution", fig2))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=avg_path, mode='lines', name='Average Path'))
    fig3.update_layout(title="Average Path (QSDE)", xaxis_title="Time", yaxis_title="Price")
    charts.append(("Average Path", fig3))

    error_pct = ((np.mean(final_prices) - S0) / S0) * 100

    summary = (
        f"Expected Final Price: ${np.mean(final_prices):.2f}\n"
        f"Std Dev: ${np.std(final_prices):.2f}\n"
        f"Option Price: ${option_price:.2f}\n"
        f"P&L Volatility: ${np.std(pnl):.2f}\n"
        f"% Error from Initial Price: {error_pct:.2f}%"
    )

    return {"charts": charts, "summary": summary}
