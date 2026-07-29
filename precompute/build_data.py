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
