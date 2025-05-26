import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from classic import generate_classical_surface, classical_simulation_results
from qsde import generate_qsde_surface, qsde_simulation_results
from replay import save_run, load_history

# --- Page Setup ---
st.set_page_config(page_title="Quantum Option Simulator", layout="wide")
st.title("Quantum vs Classical Option Pricing Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
st.sidebar.markdown("#### 🔎 Market Ticker Lookup")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
use_real = st.sidebar.checkbox("Use real market data (Yahoo Finance)", value=False)

# --- Default inputs ---
S0 = 100.0
K = 100.0
T_days = 90
sigma = 0.3
r = 0.02
div_yield = 0.00

# --- Fetch Real Market Data If Requested ---
if use_real and ticker:
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1y")
        if not hist.empty:
            S0_real = hist["Close"].iloc[-1]
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            sigma_real = log_ret.std() * np.sqrt(252)
            info = data.info
            r_real = info.get("riskFreeRate", r)
            q_real = info.get("dividendYield", div_yield)

            # Apply overrides
            S0 = float(f"{S0_real:.2f}")
            sigma = float(f"{sigma_real:.2f}")
            r = float(f"{r_real:.4f}")
            div_yield = float(f"{q_real:.4f}") if q_real else div_yield

            st.sidebar.success("Market data loaded.")
            st.sidebar.markdown(f"**S₀:** ${S0:.2f}")
            st.sidebar.markdown(f"**σ (Hist):** {sigma:.2%}")
            st.sidebar.markdown(f"**r (Risk-Free):** {r:.2%}")
            st.sidebar.markdown(f"**q (Dividend Yield):** {div_yield:.2%}")
        else:
            st.sidebar.error("No historical data found.")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")

# --- Let user still customize parameters manually ---
S0 = st.sidebar.number_input("Initial Stock Price ($)", value=S0)
K = st.sidebar.number_input("Strike Price ($)", value=K)
T_days = st.sidebar.slider("Days to Expiry", min_value=30, max_value=365, value=T_days, step=15)
T = T_days / 365
sigma = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, sigma, step=0.01)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, r, step=0.005)
div_yield = st.sidebar.slider("Dividend Yield (q)", 0.0, 0.1, div_yield, step=0.005)

# --- Navigation ---
page = st.radio("Choose Page", ["3D Volatility Surfaces", "Simulation & Results"])

# --- Page 1: Volatility Surfaces ---
if page == "3D Volatility Surfaces":
    st.subheader(" Interactive Volatility Surface Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Classical GBM Surface")
        surface_classical = generate_classical_surface(S0, T, r, sigma, div_yield)
        st.plotly_chart(surface_classical, use_container_width=True)

    with col2:
        st.markdown("### Quantum-Enhanced QSDE Surface")
        surface_qsde = generate_qsde_surface(S0, T, r, sigma, div_yield)
        st.plotly_chart(surface_qsde, use_container_width=True)

# --- Page 2: Simulation Results ---
elif page == "Simulation & Results":
    st.subheader("🔬 Full Model Simulation and Analysis")
    n_simulations = st.number_input("Number of Simulations", value=1000, step=500)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Classical GBM")
        classical_outputs = classical_simulation_results(S0, K, T, r, sigma, n_simulations)
        for label, chart in classical_outputs['charts']:
            st.markdown(f"**{label}**")
            st.plotly_chart(chart, use_container_width=True)
        st.code(classical_outputs['summary'], language='text')

    with col2:
        st.markdown("### Quantum-Enhanced QSDE")
        qsde_outputs = qsde_simulation_results(S0, K, T, r, sigma, n_simulations)
        for label, chart in qsde_outputs['charts']:
            st.markdown(f"**{label}**")
            st.plotly_chart(chart, use_container_width=True)
        st.code(qsde_outputs['summary'], language='text')

    # --- Save Current Run ---
    if st.button("Save This Run"):
        save_run("Classical GBM", 
                 {"S0": S0, "K": K, "T": T, "σ": sigma, "r": r, "n_sim": n_simulations},
                 classical_outputs["summary"],
                 None)

        save_run("QSDE", 
                 {"S0": S0, "K": K, "T": T, "σ": sigma, "r": r, "n_sim": n_simulations},
                 qsde_outputs["summary"],
                 None)

        st.success("Run saved successfully!")

    # --- Replay Section ---
    st.markdown("### 📂 Replay Previous Run")
    history = load_history()

    if history:
        selected = st.selectbox("Select a saved run:", history, format_func=lambda x: f"{x['timestamp']} — {x['model']}")
        if selected:
            st.markdown(f"#### 🔁 Replay: {selected['model']} ({selected['timestamp']})")
            st.json(selected["params"])
            st.code(selected["summary"], language="text")
            st.info("✅ Plots from replayed run coming soon!")
    else:
        st.info("No saved runs available. Click 'Save This Run' above to begin tracking your simulations.")
