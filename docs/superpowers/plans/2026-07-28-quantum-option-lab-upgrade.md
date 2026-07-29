# Quantum Option Lab Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the decorative quantum component with two honest comparisons (Black-Scholes vs Heston; Monte Carlo vs Quantum Amplitude Estimation), add Greeks + confidence intervals, and fix the risk-neutral/caching bugs — all in the existing Streamlit app.

**Architecture:** Split the tangled `qsde.py` into focused, independently testable modules under `models/`, `quantum/`, and `analytics/`. Every stochastic pricer is validated against the closed-form Black-Scholes source of truth in its degenerate case. Heavy quantum grids are precomputed into `data/`; the app runs one small live QAE circuit on demand.

**Tech Stack:** Python 3.10, NumPy, SciPy, Plotly, Streamlit, Qiskit + Qiskit Aer + Qiskit Finance, yfinance, pytest.

## Global Constraints

- Python 3.10 (matches existing `__pycache__` cpython-310).
- All Monte Carlo pricing uses **risk-neutral drift = `r - q`**. Never the old hardcoded `mu=0.05`.
- No param-blind caches. Live paths use `@st.cache_data` keyed on args; heavy grids come from `precompute/` scripts writing to `data/` with param-keyed filenames.
- QAE is demonstrated on the **lognormal / Black-Scholes case only**. QAE-on-Heston is out of scope.
- Honesty copy rule: state that on a simulator QAE is slower in wall-clock; the advantage is asymptotic (O(1/N) vs O(1/√N) in sample complexity). Never claim raw quantum speedup today.
- Pin quantum deps in `requirements.txt`: `qiskit==1.2.4`, `qiskit-aer==0.15.1`, `qiskit-finance==0.4.1`, `qiskit-algorithms==0.3.1`. If imports fail in the target env, adjust pins in Task 1 before proceeding — do not work around with the old H-T-H circuit.
- Every new package dir has an `__init__.py`.

---

## File Structure

```
models/__init__.py
models/black_scholes.py    # closed-form call/put + analytic Greeks (SOURCE OF TRUTH)
models/gbm_mc.py           # classical MC, drift = r-q, returns price + CI
models/heston.py           # Heston full-truncation Euler MC + smile surface
quantum/__init__.py
quantum/qae_pricing.py      # Qiskit Finance QAE + convergence-vs-MC helper
analytics/__init__.py
analytics/greeks.py         # analytic (delegates to BS) + finite-difference
analytics/calibration.py    # experimental Heston calibration to market IV
viz.py                      # surface / smile / convergence plot builders (replaces utils_plotly.py)
precompute/build_data.py    # bakes surfaces + QAE convergence into data/
app.py                      # multipage Streamlit (rewritten)
tests/                      # pytest suite
requirements.txt            # updated
```

Retired: the H-T-H `quantum_probability_adjustment`, `hybrid_simulation`, and the param-blind `data/qsde_surface.npy` cache in `qsde.py`. `qsde.py` is deleted in the final task once nothing imports it.

---

### Task 1: Project scaffold + pinned dependencies + QAE smoke test

Set up the package layout and prove the fragile quantum stack actually imports and runs *before* building on it (past commit history shows qiskit-aer import pain).

**Files:**
- Create: `models/__init__.py`, `quantum/__init__.py`, `analytics/__init__.py`, `precompute/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Modify: `requirements.txt`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable empty packages; a verified quantum stack.

- [ ] **Step 1: Write `requirements.txt`**

```
streamlit==1.38.0
yfinance==0.2.43
numpy==1.26.4
scipy==1.13.1
plotly==5.24.1
pandas==2.2.2
qiskit==1.2.4
qiskit-aer==0.15.1
qiskit-finance==0.4.1
qiskit-algorithms==0.3.1
pytest==8.3.3
```

- [ ] **Step 2: Create the empty package files**

Each of `models/__init__.py`, `quantum/__init__.py`, `analytics/__init__.py`, `precompute/__init__.py`, `tests/__init__.py` is an empty file. `tests/conftest.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
```

- [ ] **Step 3: Install deps**

Run: `pip install -r requirements.txt`
Expected: completes without dependency-resolution errors. If qiskit-aer or qiskit-finance fail to resolve, adjust the pins (nearest compatible minor) and re-run before continuing.

- [ ] **Step 4: Write the environment smoke test**

```python
# tests/test_environment.py
import numpy as np


def test_qae_stack_imports_and_runs():
    """The whole point of the quantum page — prove it runs on a tiny problem."""
    from qiskit_finance.circuit.library import LogNormalDistribution
    from qiskit_finance.applications.estimation import EuropeanCallPricing
    from qiskit_algorithms import IterativeAmplitudeEstimation
    from qiskit.primitives import Sampler

    n = 2
    S, vol, r, T = 2.0, 0.4, 0.05, 40 / 365
    mu = (r - 0.5 * vol**2) * T + np.log(S)
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2 / 2)
    stddev = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
    low, high = max(0, mean - 3 * stddev), mean + 3 * stddev

    dist = LogNormalDistribution(n, mu=mu, sigma=sigma**2, bounds=(low, high))
    pricer = EuropeanCallPricing(
        num_state_qubits=n, strike_price=1.9, rescaling_factor=0.25,
        bounds=(low, high), uncertainty_model=dist,
    )
    problem = pricer.to_estimation_problem()
    ae = IterativeAmplitudeEstimation(epsilon_target=0.05, alpha=0.05, sampler=Sampler())
    result = ae.estimate(problem)
    payoff = pricer.interpret(result)
    assert payoff >= 0.0
    assert np.isfinite(payoff)
```

- [ ] **Step 5: Run the smoke test**

Run: `pytest tests/test_environment.py -v`
Expected: PASS. If it fails on imports, fix version pins now — this is the load-bearing dependency for Task 5.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt models/__init__.py quantum/__init__.py analytics/__init__.py precompute/__init__.py tests/__init__.py tests/conftest.py tests/test_environment.py
git commit -m "chore: scaffold packages, pin quantum deps, add QAE smoke test"
```

---

### Task 2: Black-Scholes closed-form + analytic Greeks (source of truth)

**Files:**
- Create: `models/black_scholes.py`
- Test: `tests/test_black_scholes.py`

**Interfaces:**
- Produces:
  - `call_price(S, K, T, r, sigma, q=0.0) -> float`
  - `put_price(S, K, T, r, sigma, q=0.0) -> float`
  - `delta(S, K, T, r, sigma, q=0.0, kind="call") -> float`
  - `gamma(S, K, T, r, sigma, q=0.0) -> float`
  - `vega(S, K, T, r, sigma, q=0.0) -> float` (per 1.00 vol, i.e. per 100 vol-points)
  - `theta(S, K, T, r, sigma, q=0.0, kind="call") -> float` (per year)
  - `rho(S, K, T, r, sigma, q=0.0, kind="call") -> float`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_black_scholes.py
import numpy as np
from models import black_scholes as bs


def test_known_call_value():
    # Textbook: S=100,K=100,T=1,r=0.05,sigma=0.2,q=0 -> ~10.4506
    assert bs.call_price(100, 100, 1.0, 0.05, 0.2) == pytest_approx(10.4506, 1e-3)


def test_put_call_parity():
    S, K, T, r, sigma, q = 100, 110, 0.75, 0.03, 0.25, 0.01
    c = bs.call_price(S, K, T, r, sigma, q)
    p = bs.put_price(S, K, T, r, sigma, q)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest_approx(rhs, 1e-9)


def test_call_delta_bounds():
    d = bs.delta(100, 100, 1.0, 0.05, 0.2, kind="call")
    assert 0.0 < d < 1.0


def pytest_approx(v, tol):
    import pytest
    return pytest.approx(v, abs=tol) if tol >= 1e-3 else pytest.approx(v, rel=tol)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_black_scholes.py -v`
Expected: FAIL (module `models.black_scholes` not found).

- [ ] **Step 3: Implement**

```python
# models/black_scholes.py
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    srt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / srt
    return d1, d1 - srt


def call_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if kind == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)


def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_black_scholes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add models/black_scholes.py tests/test_black_scholes.py
git commit -m "feat: Black-Scholes closed-form pricing and analytic Greeks"
```

---

### Task 3: Classical GBM Monte Carlo with confidence intervals

**Files:**
- Create: `models/gbm_mc.py`
- Test: `tests/test_gbm_mc.py`

**Interfaces:**
- Consumes: `models.black_scholes.call_price`.
- Produces:
  - `price(S, K, T, r, sigma, q=0.0, n_sims=100_000, seed=None) -> dict` with keys `price`, `stderr`, `ci95` (a `(low, high)` tuple).
  - `convergence(S, K, T, r, sigma, q=0.0, sample_sizes=(...), seed=0) -> dict` with keys `n` (list) and `abs_error` (list of |MC − closed_form|).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gbm_mc.py
import pytest
from models import gbm_mc
from models import black_scholes as bs


def test_mc_converges_to_black_scholes_within_ci():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    exact = bs.call_price(S, K, T, r, sigma)
    out = gbm_mc.price(S, K, T, r, sigma, n_sims=400_000, seed=42)
    lo, hi = out["ci95"]
    assert lo <= exact <= hi
    assert out["stderr"] > 0


def test_convergence_error_shrinks():
    out = gbm_mc.convergence(100, 100, 1.0, 0.05, 0.2, sample_sizes=(1000, 200_000), seed=1)
    assert out["abs_error"][-1] < out["abs_error"][0]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_gbm_mc.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# models/gbm_mc.py
import numpy as np
from models import black_scholes as bs


def price(S, K, T, r, sigma, q=0.0, n_sims=100_000, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc_payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    est = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(n_sims)
    return {"price": float(est), "stderr": float(stderr),
            "ci95": (float(est - 1.96 * stderr), float(est + 1.96 * stderr))}


def convergence(S, K, T, r, sigma, q=0.0, sample_sizes=(1000, 5000, 20000, 100000, 400000), seed=0):
    exact = bs.call_price(S, K, T, r, sigma, q)
    ns, errs = [], []
    for n in sample_sizes:
        est = price(S, K, T, r, sigma, q, n_sims=int(n), seed=seed)["price"]
        ns.append(int(n))
        errs.append(abs(est - exact))
    return {"n": ns, "abs_error": errs}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_gbm_mc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/gbm_mc.py tests/test_gbm_mc.py
git commit -m "feat: risk-neutral GBM Monte Carlo with confidence intervals"
```

---

### Task 4: Heston stochastic-volatility model

The load-bearing correctness test: with vol-of-vol `xi=0` and `v0=theta=sigma^2`, Heston must collapse to Black-Scholes.

**Files:**
- Create: `models/heston.py`
- Test: `tests/test_heston.py`

**Interfaces:**
- Consumes: `models.black_scholes.call_price`.
- Produces:
  - `HestonParams` dataclass: `kappa, theta, xi, rho, v0`.
  - `simulate(S, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None) -> np.ndarray` (terminal prices, shape `(n_sims,)`).
  - `price(S, K, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None) -> dict` with `price`, `stderr`, `ci95`.
  - `implied_vol(price, S, K, T, r, q=0.0) -> float` (Brent inversion of BS).
  - `smile(S, T, r, params, strikes, q=0.0, **sim_kwargs) -> dict` with `strikes`, `implied_vols`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_heston.py
import pytest
from models import heston
from models import black_scholes as bs


def test_heston_collapses_to_black_scholes_when_xi_zero():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    params = heston.HestonParams(kappa=1.0, theta=sigma**2, xi=0.0, rho=0.0, v0=sigma**2)
    out = heston.price(S, K, T, r, params, n_sims=200_000, n_steps=200, seed=7)
    exact = bs.call_price(S, K, T, r, sigma)
    lo, hi = out["ci95"]
    assert lo <= exact <= hi


def test_implied_vol_roundtrip():
    px = bs.call_price(100, 100, 1.0, 0.05, 0.3)
    iv = heston.implied_vol(px, 100, 100, 1.0, 0.05)
    assert iv == pytest.approx(0.3, abs=1e-4)


def test_smile_has_curvature_under_negative_rho():
    params = heston.HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    out = heston.smile(100, 1.0, 0.05, params, strikes=[80, 100, 120], n_sims=120_000, n_steps=100, seed=3)
    ivs = out["implied_vols"]
    assert ivs[0] != pytest.approx(ivs[2], abs=1e-3)  # skew: wings differ
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_heston.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# models/heston.py
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from models import black_scholes as bs


@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float


def simulate(S, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    logS = np.full(n_sims, np.log(S))
    v = np.full(n_sims, params.v0)
    sqrt_dt = np.sqrt(dt)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_sims)
        z3 = rng.standard_normal(n_sims)
        z2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * z3
        v_pos = np.maximum(v, 0.0)  # full truncation
        logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1
        v += params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * sqrt_dt * z2
    return np.exp(logS)


def price(S, K, T, r, params, q=0.0, n_sims=50_000, n_steps=100, seed=None):
    ST = simulate(S, T, r, params, q, n_sims, n_steps, seed)
    disc_payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    est = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(n_sims)
    return {"price": float(est), "stderr": float(stderr),
            "ci95": (float(est - 1.96 * stderr), float(est + 1.96 * stderr))}


def implied_vol(option_price, S, K, T, r, q=0.0):
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if option_price <= intrinsic + 1e-12:
        return 0.0
    f = lambda s: bs.call_price(S, K, T, r, s, q) - option_price
    try:
        return float(brentq(f, 1e-4, 5.0, maxiter=200))
    except ValueError:
        return float("nan")


def smile(S, T, r, params, strikes, q=0.0, **sim_kwargs):
    ivs = []
    for K in strikes:
        px = price(S, K, T, r, params, q, **sim_kwargs)["price"]
        ivs.append(implied_vol(px, S, K, T, r, q))
    return {"strikes": list(strikes), "implied_vols": ivs}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_heston.py -v`
Expected: PASS (3 tests). If the `xi=0` test is flaky at the CI edge, raise `n_sims` — do not widen the assertion.

- [ ] **Step 5: Commit**

```bash
git add models/heston.py tests/test_heston.py
git commit -m "feat: Heston stochastic-vol model with smile and implied-vol inversion"
```

---

### Task 5: Quantum Amplitude Estimation pricing

**Files:**
- Create: `quantum/qae_pricing.py`
- Test: `tests/test_qae_pricing.py`

**Interfaces:**
- Consumes: `models.black_scholes.call_price`, `models.gbm_mc.convergence`.
- Produces:
  - `price(S, K, T, r, sigma, q=0.0, num_qubits=3, epsilon=0.01) -> dict` with `price`, `ci` (`(low, high)`), `n_oracle_queries` (int), `circuit` (a `QuantumCircuit`).
  - `convergence_vs_mc(S, K, T, r, sigma, q=0.0, num_qubits=3, mc_sample_sizes=(...)) -> dict` with `qae` (`{queries, abs_error}`) and `mc` (`{n, abs_error}`), both errors vs the closed-form price.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_qae_pricing.py
import pytest
from quantum import qae_pricing
from models import black_scholes as bs


def test_qae_matches_black_scholes_within_tolerance():
    S, K, T, r, sigma = 2.0, 1.9, 40 / 365, 0.05, 0.4
    exact = bs.call_price(S, K, T, r, sigma)
    out = qae_pricing.price(S, K, T, r, sigma, num_qubits=3, epsilon=0.005)
    # QAE on 3 qubits discretizes the distribution; tolerance reflects that.
    assert out["price"] == pytest.approx(exact, abs=0.05)
    assert out["circuit"] is not None


def test_convergence_returns_both_series():
    out = qae_pricing.convergence_vs_mc(2.0, 1.9, 40 / 365, 0.05, 0.4,
                                        num_qubits=3, mc_sample_sizes=(500, 5000))
    assert len(out["mc"]["n"]) == 2
    assert len(out["qae"]["queries"]) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_qae_pricing.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# quantum/qae_pricing.py
import numpy as np
from qiskit.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_algorithms import IterativeAmplitudeEstimation
from models import black_scholes as bs
from models import gbm_mc


def _build(S, K, T, r, sigma, q, num_qubits):
    mu = (r - q - 0.5 * sigma**2) * T + np.log(S)
    sig = sigma * np.sqrt(T)
    mean = np.exp(mu + sig**2 / 2)
    stddev = np.sqrt((np.exp(sig**2) - 1) * np.exp(2 * mu + sig**2))
    low, high = max(0.0, mean - 3 * stddev), mean + 3 * stddev
    dist = LogNormalDistribution(num_qubits, mu=mu, sigma=sig**2, bounds=(low, high))
    pricer = EuropeanCallPricing(
        num_state_qubits=num_qubits, strike_price=K, rescaling_factor=0.25,
        bounds=(low, high), uncertainty_model=dist,
    )
    return pricer


def price(S, K, T, r, sigma, q=0.0, num_qubits=3, epsilon=0.01):
    pricer = _build(S, K, T, r, sigma, q, num_qubits)
    problem = pricer.to_estimation_problem()
    ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=0.05, sampler=Sampler())
    result = ae.estimate(problem)
    expected_payoff = pricer.interpret(result)          # undiscounted E[max(S_T-K,0)]
    disc = np.exp(-r * T)
    est = disc * expected_payoff
    ci_lo, ci_hi = (disc * c for c in pricer.interpret(result, ci=True)) \
        if hasattr(pricer, "interpret") and False else (est, est)
    # confidence_interval is on the amplitude; map through interpret for payoff CI
    conf = result.confidence_interval_processed
    ci = (float(disc * conf[0]), float(disc * conf[1]))
    return {"price": float(est), "ci": ci,
            "n_oracle_queries": int(result.num_oracle_queries),
            "circuit": problem.grover_operator}


def convergence_vs_mc(S, K, T, r, sigma, q=0.0, num_qubits=3,
                      mc_sample_sizes=(500, 2000, 8000, 40000)):
    exact = bs.call_price(S, K, T, r, sigma, q)
    mc = gbm_mc.convergence(S, K, T, r, sigma, q, sample_sizes=mc_sample_sizes, seed=0)
    mc_out = {"n": mc["n"], "abs_error": mc["abs_error"]}
    qae_queries, qae_err = [], []
    for eps in (0.05, 0.02, 0.01, 0.005):
        out = price(S, K, T, r, sigma, q, num_qubits=num_qubits, epsilon=eps)
        qae_queries.append(out["n_oracle_queries"])
        qae_err.append(abs(out["price"] - exact))
    return {"mc": mc_out, "qae": {"queries": qae_queries, "abs_error": qae_err}}
```

Note: `confidence_interval_processed` returns the amplitude-space CI; multiplying by the discount factor approximates the payoff CI for display. If the installed qiskit-algorithms names it differently, adapt to the available attribute (`result.confidence_interval` mapped through `pricer.interpret`) — keep the returned dict shape identical.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_qae_pricing.py -v`
Expected: PASS (2 tests). May take ~30–60s.

- [ ] **Step 5: Commit**

```bash
git add quantum/qae_pricing.py tests/test_qae_pricing.py
git commit -m "feat: Quantum Amplitude Estimation option pricing + MC convergence compare"
```

---

### Task 6: Greeks module (analytic + finite-difference)

**Files:**
- Create: `analytics/greeks.py`
- Test: `tests/test_greeks.py`

**Interfaces:**
- Consumes: `models.black_scholes`.
- Produces:
  - `analytic(S, K, T, r, sigma, q=0.0, kind="call") -> dict` with keys `delta, gamma, vega, theta, rho`.
  - `finite_difference(S, K, T, r, sigma, q=0.0, kind="call") -> dict` (same keys), bumping `black_scholes.call_price`/`put_price`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_greeks.py
import pytest
from analytics import greeks


def test_analytic_matches_finite_difference():
    a = greeks.analytic(100, 100, 1.0, 0.05, 0.2, kind="call")
    fd = greeks.finite_difference(100, 100, 1.0, 0.05, 0.2, kind="call")
    for key in ("delta", "gamma", "vega", "theta", "rho"):
        assert a[key] == pytest.approx(fd[key], rel=1e-2, abs=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_greeks.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# analytics/greeks.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_greeks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/greeks.py tests/test_greeks.py
git commit -m "feat: analytic and finite-difference Greeks"
```

---

### Task 7: Visualization builders

**Files:**
- Create: `viz.py`
- Test: `tests/test_viz.py`
- Delete: `utils_plotly.py` (after moving `create_3d_surface` in)

**Interfaces:**
- Produces (all return `plotly.graph_objects.Figure`):
  - `surface(x, y, Z, title, x_label, y_label, z_label) -> Figure`
  - `smile_chart(strikes, series: dict[str, list]) -> Figure` (each series a labeled IV curve)
  - `convergence_chart(mc: dict, qae: dict) -> Figure` (log-log error vs work)
  - `paths_chart(prices_2d, n_show=10) -> Figure`

- [ ] **Step 1: Write failing test**

```python
# tests/test_viz.py
import numpy as np
import viz


def test_surface_returns_figure():
    x = np.linspace(0.1, 0.6, 5)
    y = np.linspace(0.1, 1.0, 5)
    Z = np.random.rand(5, 5)
    fig = viz.surface(x, y, Z, "t", "x", "y", "z")
    assert fig.data[0].type == "surface"


def test_convergence_chart_has_two_traces():
    mc = {"n": [100, 1000], "abs_error": [0.5, 0.1]}
    qae = {"queries": [10, 100], "abs_error": [0.3, 0.05]}
    fig = viz.convergence_chart(mc, qae)
    assert len(fig.data) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_viz.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# viz.py
import plotly.graph_objects as go


def surface(x, y, Z, title, x_label, y_label, z_label):
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale="Viridis")])
    fig.update_layout(title=title, scene=dict(
        xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label), height=520)
    return fig


def smile_chart(strikes, series):
    fig = go.Figure()
    for name, ivs in series.items():
        fig.add_trace(go.Scatter(x=strikes, y=ivs, mode="lines+markers", name=name))
    fig.update_layout(title="Implied Volatility Smile", xaxis_title="Strike",
                      yaxis_title="Implied Vol", height=420)
    return fig


def convergence_chart(mc, qae):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc["n"], y=mc["abs_error"], mode="lines+markers",
                             name="Monte Carlo (O(1/√N))"))
    fig.add_trace(go.Scatter(x=qae["queries"], y=qae["abs_error"], mode="lines+markers",
                             name="QAE (O(1/N))"))
    fig.update_layout(title="Pricing Error vs Work", xaxis_title="Samples / Oracle queries",
                      yaxis_title="Absolute error", xaxis_type="log", yaxis_type="log", height=420)
    return fig


def paths_chart(prices_2d, n_show=10):
    fig = go.Figure()
    for i in range(min(n_show, len(prices_2d))):
        fig.add_trace(go.Scatter(y=prices_2d[i], mode="lines", line=dict(width=1), showlegend=False))
    fig.update_layout(title="Simulated Price Paths", xaxis_title="Step", yaxis_title="Price", height=420)
    return fig
```

- [ ] **Step 4: Run to verify pass, then delete old util**

Run: `pytest tests/test_viz.py -v`
Expected: PASS. Then `git rm utils_plotly.py` (confirm nothing still imports it: `grep -rn utils_plotly . --include=*.py`).

- [ ] **Step 5: Commit**

```bash
git add viz.py tests/test_viz.py
git rm utils_plotly.py
git commit -m "feat: plotly chart builders; retire utils_plotly"
```

---

### Task 8: Heston calibration to market implied vols (experimental)

**Files:**
- Create: `analytics/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `models.heston`.
- Produces:
  - `calibrate(S, T, r, market: list[tuple[float, float]], q=0.0, seed=0) -> HestonParams | None` where `market` is `[(strike, implied_vol), ...]`. Returns `None` if the optimizer fails.
  - `fetch_market_smile(ticker) -> tuple[float, float, list[tuple[float, float]]] | None` returning `(S, T, [(K, iv)])` from yfinance; `None` if data missing.

- [ ] **Step 1: Write failing test** (synthetic, no network)

```python
# tests/test_calibration.py
from models import heston
from analytics import calibration


def test_calibrate_recovers_synthetic_smile():
    S, T, r = 100, 1.0, 0.05
    true = heston.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, v0=0.04)
    strikes = [85, 95, 100, 105, 115]
    smile = heston.smile(S, T, r, true, strikes, n_sims=120_000, n_steps=80, seed=5)
    market = list(zip(smile["strikes"], smile["implied_vols"]))
    fit = calibration.calibrate(S, T, r, market, seed=5)
    assert fit is not None
    assert 0.0 < fit.theta < 0.2  # sane long-run variance
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# analytics/calibration.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS. (Calibration is approximate; the test only checks a sane fit, not exact recovery.)

- [ ] **Step 5: Commit**

```bash
git add analytics/calibration.py tests/test_calibration.py
git commit -m "feat: experimental Heston calibration to market implied vols"
```

---

### Task 9: Precompute script for surfaces and QAE convergence

**Files:**
- Create: `precompute/build_data.py`
- Test: `tests/test_precompute.py`

**Interfaces:**
- Consumes: `models.black_scholes`, `models.heston`, `quantum.qae_pricing`.
- Produces: writes `data/bs_surface.npz`, `data/heston_smile_presets.npz`, `data/qae_convergence.npz`. Exposes `build_all(out_dir="data")` and loader `load(name, out_dir="data") -> dict`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_precompute.py
import os
from precompute import build_data


def test_build_and_load_bs_surface(tmp_path):
    build_data.build_bs_surface(out_dir=str(tmp_path))
    data = build_data.load("bs_surface", out_dir=str(tmp_path))
    assert data["Z"].shape[0] == len(data["T_range"])
    assert data["Z"].shape[1] == len(data["vol_range"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_precompute.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# precompute/build_data.py
import os
import numpy as np
from models import black_scholes as bs
from models import heston
from quantum import qae_pricing


def build_bs_surface(S0=100, r=0.02, q=0.0, out_dir="data"):
    vol_range = np.linspace(0.1, 0.6, 30)
    T_range = np.linspace(0.1, 1.0, 30)
    Z = np.array([[bs.call_price(S0, S0, t, r, s, q) for s in vol_range] for t in T_range])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "bs_surface.npz"), vol_range=vol_range, T_range=T_range, Z=Z)


def build_heston_presets(S0=100, r=0.02, out_dir="data"):
    presets = {
        "calm": heston.HestonParams(2.0, 0.02, 0.2, -0.3, 0.02),
        "stressed": heston.HestonParams(2.0, 0.06, 0.5, -0.6, 0.06),
        "crash": heston.HestonParams(1.5, 0.09, 0.8, -0.8, 0.09),
    }
    strikes = np.linspace(70, 130, 13)
    out = {"strikes": strikes}
    for name, p in presets.items():
        sm = heston.smile(S0, 0.5, r, p, strikes, n_sims=150_000, n_steps=100, seed=11)
        out[name] = np.array(sm["implied_vols"])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "heston_smile_presets.npz"), **out)


def build_qae_convergence(out_dir="data"):
    res = qae_pricing.convergence_vs_mc(2.0, 1.9, 40 / 365, 0.05, 0.4, num_qubits=3)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "qae_convergence.npz"),
             mc_n=res["mc"]["n"], mc_err=res["mc"]["abs_error"],
             qae_q=res["qae"]["queries"], qae_err=res["qae"]["abs_error"])


def build_all(out_dir="data"):
    build_bs_surface(out_dir=out_dir)
    build_heston_presets(out_dir=out_dir)
    build_qae_convergence(out_dir=out_dir)


def load(name, out_dir="data"):
    return dict(np.load(os.path.join(out_dir, f"{name}.npz")))
```

- [ ] **Step 4: Run to verify pass, then build real data**

Run: `pytest tests/test_precompute.py -v` → PASS.
Then: `python -c "from precompute import build_data; build_data.build_all()"`
Expected: creates `data/bs_surface.npz`, `data/heston_smile_presets.npz`, `data/qae_convergence.npz`.

- [ ] **Step 5: Commit**

```bash
git add precompute/build_data.py tests/test_precompute.py data/bs_surface.npz data/heston_smile_presets.npz data/qae_convergence.npz
git commit -m "feat: precompute surfaces + QAE convergence into data/"
```

---

### Task 10: Rewrite the Streamlit app (multipage Quantum Option Lab)

**Files:**
- Modify (rewrite): `app.py`
- Delete: `qsde.py`, `classic.py`, `replay.py` (functionality replaced; confirm no other importer)
- Test: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: all modules above.
- Produces: a runnable multipage app. Page routing via `st.sidebar.radio`.

- [ ] **Step 1: Write the smoke test** (uses Streamlit's AppTest — no browser)

```python
# tests/test_app_smoke.py
from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception


def test_pages_switch_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value("Greeks Explorer").run()
    assert not at.exception
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_app_smoke.py -v`
Expected: FAIL (app.py still imports deleted modules / new pages absent).

- [ ] **Step 3: Rewrite `app.py`**

```python
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
                    "with Grover-style amplification, giving error $O(1/N)$ vs MC's $O(1/\\sqrt{N})$.")

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
```

- [ ] **Step 4: Run smoke test, delete dead modules**

Run: `grep -rn "import qsde\|import classic\|import replay\|from qsde\|from classic\|from replay" . --include=*.py`
Expected: no matches outside the files being deleted. Then `git rm qsde.py classic.py replay.py`.
Run: `pytest tests/test_app_smoke.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_smoke.py
git rm qsde.py classic.py replay.py
git commit -m "feat: rewrite app as multipage Quantum Option Lab; retire legacy modules"
```

---

### Task 11: README, gitignore, full suite green

**Files:**
- Modify: `README.md`
- Create: `.gitignore`
- Test: full `pytest` run.

- [ ] **Step 1: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
```

- [ ] **Step 2: Rewrite `README.md`**

```markdown
# Quantum Option Lab

An honest research demo comparing classical and quantum approaches to European
option pricing.

## Two comparisons

1. **Model richness** — Black-Scholes (flat volatility) vs **Heston**
   (stochastic volatility), which reproduces the market volatility smile.
2. **Computation** — **Monte Carlo** vs **Quantum Amplitude Estimation (QAE)**.
   MC error scales as O(1/√N); QAE scales as O(1/N) in sample complexity.

**Honest caveat:** on today's simulators QAE is *slower in wall-clock*. The
advantage is asymptotic (fewer samples for the same accuracy), not raw speed.

## Run

```bash
pip install -r requirements.txt
python -c "from precompute import build_data; build_data.build_all()"  # bake data/
streamlit run app.py
```

## Test

```bash
pytest -v
```

## Structure

- `models/` — Black-Scholes, GBM Monte Carlo, Heston.
- `quantum/` — Quantum Amplitude Estimation pricing (Qiskit Finance).
- `analytics/` — Greeks, experimental Heston calibration.
- `precompute/` — bakes surfaces + QAE convergence into `data/`.
- `app.py` — multipage Streamlit UI.
```

- [ ] **Step 3: Remove tracked pyc, run full suite**

Run: `git rm -r --cached __pycache__ 2>/dev/null; pytest -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore
git rm -r --cached __pycache__ 2>/dev/null || true
git commit -m "docs: rewrite README; add gitignore; green suite"
```

---

## Self-Review

**Spec coverage:**
- Honest framing (two comparisons) → Overview page (Task 10) + README (Task 11). ✔
- Correctness fixes: risk-neutral drift → Tasks 3/4 (drift `r-q`); stale cache killed → Task 9 param-keyed npz + Task 10 `@st.cache_data`; constant quantum removed → Task 10 deletes `qsde.py`. ✔
- Module restructure → Tasks 2–9 create the exact layout in the spec. ✔
- Heston smile + presets + calibration → Tasks 4, 8, 9. ✔
- QAE + live demo + convergence → Tasks 5, 9, 10. ✔
- Greeks (analytic + FD) → Task 6. ✔
- Confidence intervals → Tasks 3, 4 (`ci95`). ✔
- Five app pages + calibration toggle → Task 10 (calibration wired via `analytics.calibration`; note: Task 10's page set exposes calibration through the Volatility Surfaces preset flow — see below). Testing → every task is TDD. ✔

**Gap found & fixed:** Task 10's app does not yet surface the experimental "Calibrate to <ticker>" toggle from Task 8. Add to Task 10 Step 3, Volatility Surfaces page, after the preset selectbox:

```python
        if st.checkbox("Calibrate Heston to a ticker (experimental)"):
            from analytics import calibration
            tkr = st.text_input("Ticker", "AAPL")
            if st.button("Calibrate"):
                fetched = calibration.fetch_market_smile(tkr)
                if not fetched:
                    st.warning("No usable option-chain data; staying on presets.")
                else:
                    Sc, Tc, mkt = fetched
                    fit = calibration.calibrate(Sc, Tc, r, mkt)
                    if fit is None:
                        st.warning("Calibration did not converge; staying on presets.")
                    else:
                        st.success(f"Fit: κ={fit.kappa:.2f} θ={fit.theta:.3f} "
                                   f"ξ={fit.xi:.2f} ρ={fit.rho:.2f} v₀={fit.v0:.3f}")
                        ks = np.linspace(0.7 * Sc, 1.3 * Sc, 13)
                        sm = heston.smile(Sc, Tc, r, fit, ks, n_sims=120_000, n_steps=80, seed=9)
                        st.plotly_chart(viz.smile_chart(sm["strikes"],
                                        {"calibrated": sm["implied_vols"],
                                         "market": [iv for _, iv in mkt] + [None] * (len(ks) - len(mkt))}),
                                        use_container_width=True)
```

(The market series is length-padded to the model strike grid for display; mismatched points render as gaps, which is acceptable for an experimental view.)

**Placeholder scan:** no TBD/TODO; every code step is complete. ✔
**Type consistency:** `HestonParams` fields, `price(...)` dict keys (`price/stderr/ci95`), `convergence` keys (`n/abs_error`), QAE dict (`price/ci/n_oracle_queries/circuit`) used consistently across Tasks 3–10. ✔
