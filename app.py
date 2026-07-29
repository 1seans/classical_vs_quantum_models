import numpy as np
import streamlit as st
import plotly.graph_objects as go

from models import black_scholes as bs
from models import gbm_mc, heston
from analytics import greeks
from precompute import build_data
import viz

st.set_page_config(page_title="Quantum Option Lab", layout="wide")
st.title("Quantum Option Lab")

st.sidebar.header("Parameters")
S0 = st.sidebar.number_input("Spot S₀ ($)", value=100.0)
K = st.sidebar.number_input("Strike K ($)", value=100.0)
T = st.sidebar.slider("Days to expiry", 30, 365, 90, 15) / 365
r = st.sidebar.slider("Risk-free r", 0.0, 0.1, 0.02, 0.005)
sigma = st.sidebar.slider("Volatility σ", 0.1, 1.0, 0.3, 0.01)
q = st.sidebar.slider("Dividend yield q", 0.0, 0.1, 0.0, 0.005)

page = st.sidebar.radio("Page", [
    "Overview", "Volatility Surfaces", "Monte Carlo vs Closed-Form",
    "Quantum: QAE vs MC", "Greeks Explorer",
])


@st.cache_data(show_spinner=False)
def _load(name):
    return build_data.load(name)


if page == "Overview":
    st.subheader("Why quantum could price options faster")
    st.markdown(
        "Two honest comparisons live here:\n\n"
        "1. **Model richness** — Black-Scholes (flat vol) vs **Heston** "
        "(stochastic vol), which reproduces the market's volatility smile.\n"
        "2. **Computation** — **Monte Carlo** vs **Quantum Amplitude Estimation**. "
        "For the same target accuracy, MC error shrinks like 1/√N while QAE shrinks "
        "like 1/N in sample complexity."
    )
    st.info("Honest caveat: on today's simulators QAE is *slower in wall-clock*. "
            "The advantage is asymptotic — fewer samples for the same accuracy — not raw speed.")
    with st.expander("▸ Show the math"):
        st.latex(r"dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{1}")
        st.latex(r"dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{2},\quad d\langle W^1,W^2\rangle=\rho\,dt")
        st.markdown("QAE encodes the expected payoff as an amplitude and estimates it "
                    r"with Grover-style amplification, giving error $O(1/N)$ vs MC's $O(1/\sqrt{N})$.")

elif page == "Volatility Surfaces":
    st.subheader("Black-Scholes vs Heston")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Black-Scholes surface (flat vol)")
        d = _load("bs_surface")
        st.plotly_chart(viz.surface(d["vol_range"], d["T_range"], d["Z"],
                        "Black-Scholes Call", "σ", "T", "Call"), use_container_width=True)
    with c2:
        st.markdown("### Heston smile")
        preset = st.selectbox("Market regime", ["calm", "stressed", "crash"])
        hp = _load("heston_smile_presets")
        st.plotly_chart(viz.smile_chart(hp["strikes"], {preset: hp[preset]}),
                        use_container_width=True)
        st.caption("Flat-vol GBM cannot bend this curve; Heston can.")

        if st.checkbox("Calibrate Heston to a ticker (experimental)"):
            from analytics import calibration
            tkr = st.text_input("Ticker", "AAPL")
            st.caption("Runs Monte Carlo inside an optimizer — expect roughly a minute.")
            if st.button("Calibrate"):
                try:
                    with st.spinner(f"Fetching {tkr} option chain and calibrating Heston…"):
                        fetched = calibration.fetch_market_smile(tkr)
                        if not fetched:
                            st.warning("No usable option-chain data; staying on presets.")
                        else:
                            Sc, Tc, mkt = fetched
                            fit = calibration.calibrate(Sc, Tc, r, mkt)
                            if fit is None:
                                st.warning("Calibration did not converge; staying on presets.")
                            else:
                                ks = np.linspace(0.7 * Sc, 1.3 * Sc, 13)
                                sm = heston.smile(Sc, Tc, r, fit, ks, n_sims=120_000, n_steps=80, seed=9)
                                st.success(f"Fit: κ={fit.kappa:.2f} θ={fit.theta:.3f} "
                                           f"ξ={fit.xi:.2f} ρ={fit.rho:.2f} v₀={fit.v0:.3f}")
                                fig = viz.smile_chart(sm["strikes"], {"calibrated": sm["implied_vols"]})
                                fig.add_trace(go.Scatter(x=[k for k, _ in mkt], y=[iv for _, iv in mkt],
                                                         mode="markers", name="market (actual strikes)"))
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Calibration unavailable ({e}); staying on presets.")

elif page == "Monte Carlo vs Closed-Form":
    st.subheader("Monte Carlo convergence to Black-Scholes")
    n = st.select_slider("Simulations", [1000, 10000, 100000, 400000], value=100000)
    exact = bs.call_price(S0, K, T, r, sigma, q)
    out = gbm_mc.price(S0, K, T, r, sigma, q, n_sims=int(n), seed=42)
    lo, hi = out["ci95"]
    m1, m2 = st.columns(2)
    m1.metric("Closed-form price", f"${exact:.4f}")
    m2.metric("Monte Carlo price", f"${out['price']:.4f}", f"95% CI ±{1.96*out['stderr']:.4f}")
    conv = gbm_mc.convergence(S0, K, T, r, sigma, q, seed=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conv["n"], y=conv["abs_error"], mode="lines+markers"))
    fig.update_layout(title="|MC − closed-form| vs simulations", xaxis_type="log",
                      yaxis_type="log", xaxis_title="Simulations", yaxis_title="Abs error", height=420)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Quantum: QAE vs MC":
    st.subheader("Quantum Amplitude Estimation vs Monte Carlo")
    d = _load("qae_convergence")
    st.plotly_chart(viz.convergence_chart(
        {"n": d["mc_n"], "abs_error": d["mc_err"]},
        {"queries": d["qae_q"], "abs_error": d["qae_err"]}), use_container_width=True)
    st.markdown("**Run a real quantum circuit** (3 qubits, ~10–30s):")
    if st.button("Run QAE live @ 3 qubits"):
        from quantum import qae_pricing
        with st.spinner("Executing amplitude estimation on Aer…"):
            res = qae_pricing.price(2.0, 1.9, 40 / 365, 0.05, 0.4, num_qubits=3, epsilon=0.01)
            exact = bs.call_price(2.0, 1.9, 40 / 365, 0.05, 0.4)
        a, b = st.columns(2)
        a.metric("QAE estimate", f"{res['price']:.4f}")
        b.metric("Closed-form", f"{exact:.4f}")
        st.caption(f"Oracle queries: {res['n_oracle_queries']} · "
                   f"CI [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
    st.info("On a simulator QAE is slower in wall-clock; the O(1/N) advantage is asymptotic.")

elif page == "Greeks Explorer":
    st.subheader("Greeks — analytic vs finite-difference")
    kind = st.radio("Option type", ["call", "put"], horizontal=True)
    a = greeks.analytic(S0, K, T, r, sigma, q, kind)
    fd = greeks.finite_difference(S0, K, T, r, sigma, q, kind)
    cols = st.columns(5)
    for col, name in zip(cols, ["delta", "gamma", "vega", "theta", "rho"]):
        col.metric(name.capitalize(), f"{a[name]:.4f}", f"FD {fd[name]:.4f}")
    with st.expander("▸ What these mean"):
        st.markdown("**Delta**: ∂price/∂spot · **Gamma**: ∂delta/∂spot · "
                    "**Vega**: ∂price/∂σ · **Theta**: ∂price/∂time · **Rho**: ∂price/∂r.")
