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
