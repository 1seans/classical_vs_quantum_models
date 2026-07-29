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
