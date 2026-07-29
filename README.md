# Quantum Option Lab

An honest research demo comparing classical and quantum approaches to European
option pricing.

**Live App:** https://classicalvsquantummodels-lzxa5schzvrb7tufdtdwht.streamlit.app/

## Two comparisons

1. **Model richness** — Black-Scholes (flat volatility) vs **Heston**
   (stochastic volatility), which reproduces the market volatility smile.
2. **Computation** — **Monte Carlo** vs **Quantum Amplitude Estimation (QAE)**.
   MC error scales as O(1/√N); QAE scales as O(1/N) in sample complexity.

**Honest caveat:** on today's simulators QAE is *slower in wall-clock*. The
advantage is asymptotic (fewer samples for the same accuracy), not raw speed.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from precompute import build_data; build_data.build_all()"  # bake data/
.venv/bin/streamlit run app.py
```

## Test

```bash
.venv/bin/pytest -v
```

## Structure

- `models/` — Black-Scholes (analytic source of truth), GBM Monte Carlo, Heston.
- `quantum/` — Quantum Amplitude Estimation pricing (Qiskit Finance).
- `analytics/` — Greeks, experimental Heston calibration.
- `precompute/` — bakes surfaces + QAE convergence into `data/`.
- `viz.py` — Plotly chart builders.
- `app.py` — multipage Streamlit UI.
