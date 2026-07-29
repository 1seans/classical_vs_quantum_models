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
