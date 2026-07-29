# Quantum Option Lab — Upgrade Design

**Date:** 2026-07-28
**Repo:** `classical_vs_quantum_models` (upgrade in place, keep Streamlit + GitHub deploy)
**Status:** Approved design, pre-implementation

## Purpose

Turn the current "Quantum vs Classical Option Pricing" demo into a rigorous,
honest research module ("Quantum Option Lab") suitable for a public research
site. It replaces the current decorative quantum component with two genuine,
defensible comparisons, upgrades the model class beyond flat-vol GBM, and adds
the analytics a quant expects.

This module is self-contained but built so it can later slot in as a section of
the larger Lucy research site.

## Background: what's wrong today

- **The "quantum" part is a constant.** `quantum_probability_adjustment()`
  builds a deterministic H–T–H circuit that always returns P(0) = (2+√2)/4 ≈
  0.8536. The "quantum enhancement" reduces to a fixed +4.3% drift bump on GBM,
  computed redundantly 100× per grid cell.
- **Stale cache ignores inputs.** `generate_qsde_surface` loads
  `data/qsde_surface.npy` and returns it regardless of S0/σ/r, so the quantum
  surface never responds to the controls.
- **Risk-neutral inconsistency.** Both Monte Carlo paths drift at hardcoded
  `mu=0.05` then discount at `r`. Option-pricing MC must drift at `r`, so
  classical MC does not even match the closed-form Black-Scholes surface.

## The honest framing (backbone)

Replace the implied "quantum price process beats classical" claim with two
comparisons that are each true and measurable:

1. **Model richness — Black-Scholes/GBM vs Heston.** GBM has flat volatility and
   cannot produce a smile/skew. Heston makes variance a mean-reverting
   stochastic process, producing the smile real markets show. Demonstrates *why*
   a richer model matters.
2. **Computation method — Monte Carlo vs Quantum Amplitude Estimation (QAE).**
   The *same* pricing problem solved two ways. Classical MC error scales as
   O(1/√N); QAE scales as O(1/N) in query/sample complexity (the canonical
   Qiskit Finance result).

**Honesty guardrail (in the copy):** on a simulator, QAE is *slower in
wall-clock*. The advantage is asymptotic — fewer samples for the same accuracy —
not raw speed today. State this plainly. Do not overclaim quantum advantage.

The old "quantum drift bias" is retired. Optionally demoted to a clearly-labeled
"quantum RNG" curiosity, not part of the main narrative.

## Architecture / module restructure

Split the tangled `qsde.py` into focused, independently testable modules:

```
models/black_scholes.py   # closed-form call/put + analytic Greeks (source of truth)
models/gbm_mc.py          # classical MC, risk-neutral drift = r, with confidence intervals
models/heston.py          # Heston MC + smile surface + optional calibration
quantum/qae_pricing.py    # Qiskit Finance QAE (lognormal loader + IterativeAmplitudeEstimation)
analytics/greeks.py       # analytic + finite-difference Greeks
viz.py                    # surface / smile / convergence plots (replaces utils_plotly.py)
precompute/               # scripts that bake QAE grids + surfaces into data/
app.py                    # multipage Streamlit
```

Design rules per module:
- `black_scholes.py` is the analytic source of truth; every other pricer is
  validated against it in the degenerate cases.
- All Monte Carlo uses **risk-neutral drift = r**.
- No hand-rolled param-ignoring caches. Use `@st.cache_data` keyed on args for
  live paths; use `precompute/` scripts for the heavy QAE grids written to
  `data/` (keyed filenames or a manifest, never a single param-blind blob).

### Module contracts (summary)

- `black_scholes.call_price(S,K,T,r,sigma,q) -> float`, plus `put_price`,
  and Greeks `delta/gamma/vega/theta/rho`.
- `gbm_mc.price(S,K,T,r,sigma,q,n_sims,n_steps,seed) -> {price, stderr, ci95}`.
- `heston.simulate(...)`, `heston.price(...) -> {price, stderr, ci95}`,
  `heston.smile_surface(...) -> grid`, `heston.calibrate(chain) -> params | None`
  (full-truncation Euler scheme; Feller condition surfaced to the user).
- `qae_pricing.price(S,K,T,r,sigma,num_uncertainty_qubits) -> {estimate, ci,
  circuit, n_oracle_calls}` using `LogNormalDistribution` +
  `EuropeanCallPricing` + `IterativeAmplitudeEstimation` on `AerSimulator`.
  Includes a `convergence_vs_mc(...)` helper for precompute.
- `greeks.analytic(...)` and `greeks.finite_difference(...)`.

## App — "Quantum Option Lab" (multipage Streamlit)

1. **Overview** — layered explanation. Plain-English top ("why quantum could
   price options faster"), with ▸ *Show the math* expanders (Heston SDE, QAE
   amplitude → error O(1/N), circuit). Tells the research story.
2. **Volatility Surfaces** — Black-Scholes vs Heston side by side. Live
   smile/skew responding to κ, θ, ξ, ρ, v₀ sliders, plus calm / stressed /
   crash-risk presets.
3. **Monte Carlo vs Closed-Form** — MC price with 95% confidence interval and a
   convergence-to-Black-Scholes sanity plot (error shrinking as N grows).
4. **Quantum: QAE vs MC** — precomputed convergence comparison (error vs
   samples, MC O(1/√N) vs QAE O(1/N)), plus a **"Run QAE live @ 3 qubits"**
   button: real Aer execution showing estimate, error bars, and the circuit
   diagram (~10–30s, clearly labeled).
5. **Greeks Explorer** — delta/gamma/vega/theta/rho, analytic vs
   finite-difference.

Heston **"Calibrate to <ticker>"** is an experimental toggle (yfinance option
chain → fit κθξρv₀ to market implied vols) with graceful fallback to manual
sliders if data is missing or the fit diverges. Manual sliders are the default
and always work.

### QAE runtime strategy

- Surfaces and QAE-vs-MC convergence: **precomputed** into `data/` for smooth,
  instant browsing.
- One **live** small-qubit (3) QAE run behind a button, so visitors see a real
  circuit execute without the app hanging on every interaction.

## Audience & explanation depth

Layered for the widest reach: intuitive plain-English by default, with
expandable "Show the math" sections (Heston equations, QAE Grover operator and
error bound, Aer circuit) for quant readers. Greeks and confidence intervals
explained both ways.

## Testing (pytest)

- **Put-call parity** and known Black-Scholes values (closed-form correctness).
- **MC converges to Black-Scholes** within its reported 95% CI.
- **Heston collapses to Black-Scholes when ξ → 0** (vol-of-vol zero) — key
  correctness check that the Heston implementation is right.
- **QAE estimate matches Black-Scholes** within tolerance for the lognormal case.
- **Greeks match finite-difference** within tolerance.

## Scope guardrails (YAGNI)

- QAE is demonstrated on the canonical Black-Scholes/lognormal case only.
  **QAE-on-Heston is explicitly out of scope** (research-grade distribution
  loading) — an optional future stretch, not a dependency.
- No new web framework. Upgrade the existing Streamlit app in place; keep the
  GitHub → Streamlit Cloud deploy.
- Lucy research-site integration is a later project; this module is built to
  slot in but does not implement it here.

## Deliverables

- Restructured modules per the layout above, with the three correctness fixes.
- Precompute scripts + baked `data/` artifacts for surfaces and QAE convergence.
- Multipage Streamlit app with the five pages + calibration toggle.
- pytest suite covering the checks above.
- Updated `requirements.txt` (adds `qiskit-finance`; pins compatible
  `qiskit`/`qiskit-aer`) and README explaining the honest framing.
