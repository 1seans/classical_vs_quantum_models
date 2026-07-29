import numpy as np
try:
    from qiskit.primitives import Sampler  # qiskit 1.x (V1 primitive)
except ImportError:  # qiskit 2.x removed the V1 Sampler
    from qiskit.primitives import StatevectorSampler as Sampler
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
    conf = result.confidence_interval_processed         # payoff-space CI (post-processed)
    ci = (float(disc * conf[0]), float(disc * conf[1]))
    return {"price": float(est), "ci": ci,
            "n_oracle_queries": int(result.num_oracle_queries),
            "circuit": problem.state_preparation}


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
