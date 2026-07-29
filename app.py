import numpy as np
import streamlit as st
import plotly.graph_objects as go

from models import black_scholes as bs
from models import gbm_mc, heston
from analytics import greeks
from precompute import build_data
import viz

st.set_page_config(page_title="Quantum Option Lab", page_icon="◈", layout="wide")

# ----------------------------------------------------------------------------
# Quantum-lab theme (deep navy, cyan -> violet accents, glass cards, Syne type)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Archivo:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --q-cyan:#22d3ee; --q-violet:#c084fc; --q-indigo:#818cf8;
  --q-bg:#0a0e1a; --q-panel:#121829; --q-text:#e6edf7; --q-muted:#8b98b8;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 620px at 82% -12%, rgba(34,211,238,0.12), transparent 60%),
    radial-gradient(1000px 560px at -8% 8%, rgba(192,132,252,0.10), transparent 55%),
    var(--q-bg);
}
[data-testid="stHeader"]{ background:transparent; }

html, body, [data-testid="stAppViewContainer"], [class*="css"]{
  font-family:'Archivo', sans-serif; color:var(--q-text);
}
h1,h2,h3,h4{ font-family:'Syne', sans-serif !important; letter-spacing:-0.02em; }

/* Hero */
.q-hero{ padding:8px 0 6px; }
.q-eyebrow{
  font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.28em;
  color:var(--q-cyan); text-transform:uppercase; margin-bottom:10px; opacity:0.9;
}
.q-title{
  font-family:'Syne', sans-serif; font-weight:800; font-size:clamp(38px,6vw,66px);
  line-height:0.98; margin:0;
  background:linear-gradient(100deg,#67e8f9 0%, var(--q-cyan) 35%, var(--q-indigo) 70%, var(--q-violet) 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.q-tag{ color:var(--q-muted); font-size:17px; max-width:720px; margin:14px 0 2px; }
.q-rule{ height:1px; margin:18px 0 4px;
  background:linear-gradient(90deg, rgba(34,211,238,0.6), rgba(192,132,252,0.3), transparent); }

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0c1120 0%, #0a0e1a 100%);
  border-right:1px solid rgba(34,211,238,0.12);
}
[data-testid="stSidebar"] h2{ font-size:15px; letter-spacing:0.14em; text-transform:uppercase; color:var(--q-muted) !important; }

/* Metric cards */
[data-testid="stMetric"]{
  background:rgba(18,24,41,0.72); border:1px solid rgba(34,211,238,0.18);
  border-radius:16px; padding:16px 18px;
  box-shadow:0 10px 34px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
}
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono', monospace; color:var(--q-cyan); }
[data-testid="stMetricLabel"]{ color:var(--q-muted); }

/* Expanders as glass cards (used by the "What am I looking at?" explainers) */
[data-testid="stExpander"]{
  border:1px solid rgba(129,140,248,0.28); border-radius:16px;
  background:rgba(18,24,41,0.55); backdrop-filter:blur(6px); overflow:hidden;
}
[data-testid="stExpander"] summary{ font-family:'Syne', sans-serif; font-weight:600; }
[data-testid="stExpander"] summary:hover{ color:var(--q-cyan); }

/* Buttons */
.stButton>button{
  background:linear-gradient(100deg, var(--q-cyan), var(--q-violet)); color:#07101f;
  border:none; border-radius:12px; font-weight:600; padding:8px 18px;
  box-shadow:0 8px 24px rgba(34,211,238,0.22); transition:all .15s ease;
}
.stButton>button:hover{ filter:brightness(1.08); transform:translateY(-1px);
  box-shadow:0 12px 30px rgba(192,132,252,0.30); }

/* Alerts (st.info / st.warning) softened to match */
[data-testid="stAlert"]{ border-radius:14px; }

/* Section subheads */
h2, h3{ margin-top:6px; }
.q-foot{ color:var(--q-muted); font-size:13px; text-align:center; padding:26px 0 8px;
  font-family:'IBM Plex Mono', monospace; letter-spacing:0.04em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="q-hero">
  <div class="q-eyebrow">◈ Quantum Finance · Research Lab</div>
  <div class="q-title">Quantum Option Lab</div>
  <p class="q-tag">Pricing options two honest ways — classical vs quantum — and showing where each one actually wins. Move the sliders on the left; everything reprices live.</p>
</div>
<div class="q-rule"></div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
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


def explain(body):
    """Plain-English 'what am I looking at' expander for non-technical visitors."""
    with st.expander("🔍  New here? What am I looking at? (plain English)"):
        st.markdown(body)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Why quantum could price options faster")
    explain(
        "**An option is a bet on where a price ends up.** To price that bet fairly, "
        "you have to imagine thousands of possible futures for the stock and average what the "
        "bet would pay in each one.\n\n"
        "This site does that two ways and compares them honestly:\n"
        "- A **classic** method (Monte Carlo — just simulate lots of random futures), and\n"
        "- A **quantum** method (Amplitude Estimation — a quantum computer can reach the same "
        "accuracy with far fewer tries).\n\n"
        "It also shows a smarter market model (**Heston**) that captures how real markets get "
        "calmer and wilder over time. **Try it:** drag the sliders on the left and watch the "
        "numbers and charts update."
    )
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
    explain(
        "A **volatility surface** is a 3D price map. It shows what an option costs as you change "
        "two things at once: how jumpy the stock is (**volatility**, left-right) and how long until "
        "the bet expires (**time**, front-back). Height is the option's price.\n\n"
        "- The **smooth sheet** is the textbook *Black-Scholes* model — clean, but it assumes jumpiness "
        "never changes.\n"
        "- The **rippled sheet** is the *Heston* model, where the jumpiness itself drifts over time "
        "(more like real markets). The bumpy texture is the fingerprint of running thousands of random "
        "simulations — that's what 'Monte Carlo' looks like.\n\n"
        "Below, the **smile** shows the same idea in 2D: real markets charge more for far-out bets, "
        "bending the line into a smile that flat-vol models can't produce."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Black-Scholes surface")
        st.caption("Flat volatility · exact closed-form · perfectly smooth")
        d = _load("bs_surface")
        st.plotly_chart(viz.surface(d["vol_range"], d["T_range"], d["Z"],
                        "Black-Scholes Call", "σ", "T", "Call price"), use_container_width=True)
    with c2:
        st.markdown("### Heston surface")
        st.caption("Stochastic volatility · Monte Carlo · real sampling texture")
        hs = _load("heston_surface")
        st.plotly_chart(viz.surface(hs["vol_range"], hs["T_range"], hs["Z"],
                        "Heston Call", "σ (initial)", "T", "Call price"), use_container_width=True)

    st.markdown("### The volatility smile")
    preset = st.selectbox("Market regime", ["calm", "stressed", "crash"])
    hp = _load("heston_smile_presets")
    st.plotly_chart(viz.smile_chart(hp["strikes"], {preset: hp[preset]}), use_container_width=True)
    st.caption("Flat-vol Black-Scholes would draw a straight line here; Heston bends it into the smile real markets show.")

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
    explain(
        "**Monte Carlo** is the 'just try it a lot' method: simulate thousands of random price "
        "futures, see what the option pays in each, and average. The more futures you simulate, the "
        "closer your estimate gets to the exact answer.\n\n"
        "The **closed-form price** is that exact textbook answer we're aiming for. Crank up the "
        "**Simulations** slider and watch the two numbers converge — and the error curve slide down. "
        "The '95% CI' is an honesty band: the true price is almost certainly inside it."
    )
    n = st.select_slider("Simulations", [1000, 10000, 100000, 400000], value=100000)
    exact = bs.call_price(S0, K, T, r, sigma, q)
    out = gbm_mc.price(S0, K, T, r, sigma, q, n_sims=int(n), seed=42)
    m1, m2 = st.columns(2)
    m1.metric("Closed-form price", f"${exact:.4f}")
    m2.metric("Monte Carlo price", f"${out['price']:.4f}", f"95% CI ±{1.96*out['stderr']:.4f}")
    conv = gbm_mc.convergence(S0, K, T, r, sigma, q, seed=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conv["n"], y=conv["abs_error"], mode="lines+markers",
                             line=dict(color=viz.CYAN, width=3), marker=dict(size=7)))
    viz._theme(fig)
    fig.update_layout(title="|MC − closed-form| vs simulations", xaxis_type="log",
                      yaxis_type="log", xaxis_title="Simulations", yaxis_title="Absolute error")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Quantum: QAE vs MC":
    st.subheader("Quantum Amplitude Estimation vs Monte Carlo")
    explain(
        "Both methods estimate the **same** option price — the question is how much *work* each needs "
        "to hit a given accuracy.\n\n"
        "- **Monte Carlo** has to *quadruple* its samples to halve its error.\n"
        "- **Quantum Amplitude Estimation (QAE)** only needs to *double* its work to halve its error — "
        "that's the theoretical quantum speedup, and it's what the chart shows.\n\n"
        "**Honest caveat:** on today's quantum *simulators* this is actually slower in real seconds. "
        "The win is about how the error shrinks as problems get bigger, not raw speed today. Hit the "
        "button to run a real (tiny, 3-qubit) quantum circuit and see it estimate the price live."
    )
    d = _load("qae_convergence")
    st.plotly_chart(viz.convergence_chart(
        {"n": d["mc_n"], "abs_error": d["mc_err"]},
        {"queries": d["qae_q"], "abs_error": d["qae_err"]}), use_container_width=True)
    st.markdown("**Run a real quantum circuit** (3 qubits, ~10–30s):")
    if st.button("Run QAE live @ 3 qubits"):
        try:
            from quantum import qae_pricing
            with st.spinner("Executing amplitude estimation on Aer…"):
                res = qae_pricing.price(2.0, 1.9, 40 / 365, 0.05, 0.4, num_qubits=3, epsilon=0.01)
                exact = bs.call_price(2.0, 1.9, 40 / 365, 0.05, 0.4)
            a, b = st.columns(2)
            a.metric("QAE estimate", f"{res['price']:.4f}")
            b.metric("Closed-form", f"{exact:.4f}")
            st.caption(f"Oracle queries: {res['n_oracle_queries']} · "
                       f"CI [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
        except Exception as e:
            st.warning(f"Live quantum demo unavailable in this environment ({e}). "
                       "The precomputed QAE-vs-MC comparison above still reflects real runs.")
    st.info("On a simulator QAE is slower in wall-clock; the O(1/N) advantage is asymptotic.")

elif page == "Greeks Explorer":
    st.subheader("Greeks — analytic vs finite-difference")
    explain(
        "The **Greeks** measure how twitchy an option's price is to the things around it — traders "
        "watch them to manage risk:\n\n"
        "- **Delta** — moves when the stock moves\n"
        "- **Gamma** — how fast Delta itself changes\n"
        "- **Vega** — sensitivity to volatility\n"
        "- **Theta** — value lost as time ticks by\n"
        "- **Rho** — sensitivity to interest rates\n\n"
        "Each card shows the exact formula value, with a second number ('FD') computed a completely "
        "different way as a sanity check. They should match — and they do."
    )
    kind = st.radio("Option type", ["call", "put"], horizontal=True)
    a = greeks.analytic(S0, K, T, r, sigma, q, kind)
    fd = greeks.finite_difference(S0, K, T, r, sigma, q, kind)
    cols = st.columns(5)
    for col, name in zip(cols, ["delta", "gamma", "vega", "theta", "rho"]):
        col.metric(name.capitalize(), f"{a[name]:.4f}", f"FD {fd[name]:.4f}", delta_color="off")

st.markdown('<div class="q-foot">Quantum Option Lab · Black-Scholes · Heston · Quantum Amplitude Estimation</div>',
            unsafe_allow_html=True)
