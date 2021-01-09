import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.manifold import TSNE

from scipy.sparse.csgraph import connected_components
from umap import UMAP


def plot_learning_curve(filename, epochs, stack=None, history=None, dark=False):
    if stack is None and history is None:
        return

    if stack is not None:
        history = stack.history

    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    e = range(epochs)
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    sns.lineplot(e, history['accuracy'], label="accuracy", color='darkcyan')
    sns.lineplot(e, history['val_accuracy'], label='val_accuracy', color='coral')
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    sns.lineplot(e, history['loss'], label="loss", color='darkcyan')
    sns.lineplot(e, history['val_loss'], label='val_loss', color='coral')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    plt.savefig(filename)
    plt.show()
    plt.figure()


def plot_feature_map(layer_outputs, y, label_list, filename=None, fig_len=10, umap=False, dark=False):
    compressor = TSNE(n_components=2)
    if umap:
        compressor = UMAP(n_components=2)

    x_cmprss = compressor.fit_transform(layer_outputs)

    x_cmprss = pd.DataFrame(x_cmprss, columns=["x", "y"])
    x_cmprss["label"] = [label_list[y_] for y_ in y]

    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.figure(figsize=(fig_len, fig_len))
    sns.lmplot(x="x", y="y", hue="label", data=x_cmprss,
               fit_reg=False, palette="Set2", x_jitter=0.2, y_jitter=0.2)
    if filename is not None:
        plt.savefig(filename)

    plt.figure()

    return x_cmprss