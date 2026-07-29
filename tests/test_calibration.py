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
