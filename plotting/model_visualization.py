from plotly import graph_objects as go, figure_factory as ff

import pandas as pd
from sklearn.manifold import TSNE


def plotly_learning_curve(history, loss=True, accuracy=True):
    data = []

    epochs = list(range(0, len(history["val_loss"])))

    if loss:
        data.append(go.Scatter(x=epochs,
                               y=history["val_loss"],
                               mode='lines',
                               name='Test loss',
                               line=dict(dash='dot')))

        data.append(go.Scatter(x=epochs,
                               y=history["loss"],
                               mode='lines',
                               name='Training loss'))

    if accuracy:
        data.append(go.Scatter(x=epochs,
                               y=history["val_accuracy"],
                               mode='lines',
                               name='Test accuracy',
                               line=dict(dash='dot')))

        data.append(go.Scatter(x=epochs,
                               y=history["accuracy"],
                               mode='lines',
                               name='Training accuracy'))

    fig = go.Figure(data=data, layout=go.Layout(xaxis={"title": "Epochs"}))

    return fig


def plotly_feature_map(layer_outputs, y, label_list):
    compressor = TSNE(n_components=2)

    x_cmprss = compressor.fit_transform(layer_outputs)

    x_cmprss = pd.DataFrame(x_cmprss, columns=["x", "y"])
    x_cmprss["label"] = [label_list[y_] for y_ in y]

    trace = []
    for class_name in label_list:
        trace.append(go.Scatter(x=x_cmprss["x"][x_cmprss["label"] == class_name],
                                y=x_cmprss["y"][x_cmprss["label"] == class_name],
                                mode='markers',
                                name=class_name))

    fig = go.Figure(data=trace)
    return fig
