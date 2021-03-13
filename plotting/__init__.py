from .model_visualization import *
from .metrics_visualization import *

import pandas as pd
import plotly.graph_objs as go


__all__ = ["plotly_learning_curve",
           "plotly_feature_map",
           "SeabornColorPalette",
           "plotly_boxplot",
           "plotly_heatmap",
           "plotly_multi_boxplot"]


def plotly_motion_sensor_data(sensor_data: pd.DataFrame, to_datetime=False):
    """plot sensor data using by Plotly

    sensor_data: pd.DataFrame `sensor_data`.columns is ["time", "x", "y", "z"]
    """
    if to_datetime:
        timestamp = pd.to_datetime(sensor_data["time"], format="%Y/%m/%d %H:%M:%S.%f")
    else:
        timestamp = sensor_data["time"]

    plot_data = []

    for axis_ in sensor_data.columns[1:]:
        plot_data.append(go.Scatter(x=timestamp,
                                    y=sensor_data[axis_],
                                    mode='lines',
                                    name=axis_))

    fig = go.Figure(data=plot_data)
    return fig
