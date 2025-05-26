# utils_plotly.py

import plotly.graph_objects as go
import numpy as np

def create_3d_surface(x, y, z, title="3D Surface", x_label="X", y_label="Y", z_label="Z"):
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
