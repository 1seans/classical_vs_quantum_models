import plotly.graph_objects as go

# Quantum-lab palette (kept in sync with app.py CSS + .streamlit/config.toml)
CYAN = "#22d3ee"
VIOLET = "#c084fc"
INDIGO = "#818cf8"
TEXT = "#e6edf7"
MUTED = "#8b98b8"

# Deep-navy -> cyan -> violet, so surfaces read as part of the same system.
QUANTUM_COLORSCALE = [
    [0.00, "#0e1b3a"],
    [0.25, "#155e75"],
    [0.50, "#22d3ee"],
    [0.75, "#818cf8"],
    [1.00, "#c084fc"],
]


def _theme(fig, height=420):
    """Apply the shared dark quantum-lab look to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Archivo, sans-serif", color=TEXT),
        title_font=dict(family="Syne, sans-serif", color=TEXT, size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=48, b=10),
        height=height,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def surface(x, y, Z, title, x_label, y_label, z_label):
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=Z, colorscale=QUANTUM_COLORSCALE,
        showscale=True, colorbar=dict(thickness=12, outlinewidth=0),
    )])
    fig.update_layout(scene=dict(
        xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
    ))
    return _theme(fig, height=520).update_layout(title=title)


def smile_chart(strikes, series):
    fig = go.Figure()
    colors = [CYAN, VIOLET, INDIGO]
    for i, (name, ivs) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=strikes, y=ivs, mode="lines+markers", name=name,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=7),
        ))
    _theme(fig)
    fig.update_layout(title="Implied Volatility Smile", xaxis_title="Strike", yaxis_title="Implied Vol")
    return fig


def convergence_chart(mc, qae):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc["n"], y=mc["abs_error"], mode="lines+markers", name="Monte Carlo  ·  O(1/√N)",
        line=dict(color=CYAN, width=3), marker=dict(size=7)))
    fig.add_trace(go.Scatter(
        x=qae["queries"], y=qae["abs_error"], mode="lines+markers", name="Quantum (QAE)  ·  O(1/N)",
        line=dict(color=VIOLET, width=3), marker=dict(size=7)))
    _theme(fig)
    fig.update_layout(
        title="Pricing Error vs Work", xaxis_title="Samples / oracle queries",
        yaxis_title="Absolute error", xaxis_type="log", yaxis_type="log")
    return fig


def paths_chart(prices_2d, n_show=10):
    fig = go.Figure()
    for i in range(min(n_show, len(prices_2d))):
        fig.add_trace(go.Scatter(y=prices_2d[i], mode="lines",
                                 line=dict(width=1, color=CYAN), opacity=0.5, showlegend=False))
    _theme(fig)
    fig.update_layout(title="Simulated Price Paths", xaxis_title="Step", yaxis_title="Price")
    return fig
