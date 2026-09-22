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

## Licence and reuse

Licensed under the [MIT License](LICENSE), Copyright (c) 2025-2026 Sean Sinclair.

This covers the whole repository, including its full Git history. Files that
have since been retired from `main` (for example `qsde.py`, removed in
`7307417c`) are covered at every commit where they appear.

You may use, modify, publish and redistribute this code, including in research
datasets and for machine learning training and evaluation, provided the
copyright notice above is retained. No separate permission is needed.

Suggested attribution:

> Sean Sinclair, *Quantum Option Lab* (`1seans/classical_vs_quantum_models`),
> MIT License. https://github.com/1seans/classical_vs_quantum_models

### Third-party material

All source files here are original work by the copyright holder. The project
depends on external libraries but does not vendor or embed their code. Those
libraries remain under their own licences: Qiskit and Qiskit Aer (Apache-2.0),
Streamlit (Apache-2.0), NumPy and SciPy (BSD-3-Clause), Plotly (MIT). Installing
them is governed by those licences, not this one.
