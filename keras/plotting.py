import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.manifold import TSNE

from scipy.sparse.csgraph import connected_components
from umap import UMAP

from tensorflow.python.util import nest
import tensorflow as tf
import pydot


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


# https://qiita.com/T-STAR/items/f3adf908a7be8e5731ff
def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=False,
                 show_params=False,
                 show_configs=False,
                 splines=True,
                 concentrate=True,
                 fillcolor='white',
                 bgcolor='white',
                 fgcolor='black',
                 horizontal=False,
                 dpi=None,
                 config_list=None,
                 colormap=None):
    """
    Extend tf.keras.utils.model_to_dot

    Parameters:
        model: A Keras model instance
        show_shapes: whether to display shape information
        show_layer_names: whether to display layer names
        show_params:
        show_configs:
        splines:
        concentrate:
        fillcolor: Color of fill
        bgcolor: Background color
        fgcolor: Figure color
        horizontal: whether to display horizontal
        dpi: Dots per inch
        config_list:
        colormap:
    """
    from tensorflow.python.keras.engine import sequential

    rankdir = 'TB' if not horizontal else 'LR'

    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('splines', splines)
    dot.set('concentrate', concentrate)
    dot.set('dpi', dpi)
    dot.set('bgcolor', bgcolor)
    dot.set_node_defaults(shape='plain', style='filled', fontcolor=fgcolor)

    if colormap is None:
        colormap = {
            'layers.merge': 'lightgray',
            'Flatten': 'lightgray',
            'Reshape': 'lightgray',
            'Lambda': 'mediumpurple',
            'BatchNormalization': 'aquamarine',
            'activations': 'lightpink',
            'Activation': 'lightpink',
            'layers.convolutional': 'lightskyblue',
            'Dense': 'lightskyblue',
            'layers.embeddings': 'khaki',
            'layers.pooling': 'orange',
            'recurrent': 'turquoise',
            'Dropout': 'khaki',
            # 'layers.core':          'lightblue',
        }
    if config_list is None:
        config_list = ['activation', 'filters', 'units', 'kernel_size', 'pool_size', 'strides', 'padding', 'use_bias',
                       'momentum', 'rate']

    model_layers = model.layers
    if not model._is_graph_network:
        node = pydot.Node(str(id(model)), label=model.name)
        dot.add_node(node)
        return dot
    elif isinstance(model, sequential.Sequential):
        if not model.built:
            model.build()
        model_layers = super(sequential.Sequential, model).layers

    # Create graph nodes.
    for i, layer in enumerate(model_layers):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__

        if isinstance(layer, tf.keras.layers.Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        header_color = fillcolor
        if class_name in colormap:
            header_color = colormap[class_name]
        else:
            for cls in colormap:
                if cls in str(type(layer)):
                    header_color = colormap[cls]
                    break

        record_list = []

        if show_params:
            params = layer.count_params()
            if params != 0:
                record_list.append(('params', f'{layer.count_params():,}'))

        if show_shapes:
            def format_shape(shape):
                return str(shape).replace(str(None), 'None')

            try:
                outputlabels = format_shape(layer.output_shape)
            except AttributeError:
                outputlabels = '?'
            if hasattr(layer, 'input_shape'):
                inputlabels = format_shape(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [format_shape(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = '?'
            record_list.append(('input', inputlabels))
            record_list.append(('output', outputlabels))

        if show_configs:
            try:
                layer_config = layer.get_config()
                record_list_ext = []
                for key in config_list:
                    if key in layer_config:
                        record_list_ext.append((key, layer_config[key]))
                record_list.extend(record_list_ext)
            except NotImplementedError:
                pass

        def make_td(name, val):
            return f'<tr><td>{name}</td><td>{val}</td></tr>'

        label = f'<<table color=\"{fgcolor}\" bgcolor=\"{fillcolor}\" border=\"0\" style=\"border-collapse: collapse\" cellborder=\"1\" cellspacing=\"0\" cellpadding=\"3\">'
        label += f'<tr ><td bgcolor=\"{header_color}\" colspan=\"2\">{class_name}</td></tr>'
        if show_layer_names:
            label += make_td('name', layer_name)
        for name, val in record_list:
            label += make_td(name, val)
        label += '</table>>'

        node = pydot.Node(name=layer_id, label=label)
        dot.add_node(node)

    def add_edge(dot, src, dst):
        if not dot.get_edge(src, dst):
            dot.add_edge(pydot.Edge(src, dst, color=fgcolor))

    for layer in model_layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in nest.flatten(node.inbound_layers):
                    inbound_layer_id = str(id(inbound_layer))
                    assert dot.get_node(inbound_layer_id)
                    assert dot.get_node(layer_id)
                    add_edge(dot, inbound_layer_id, layer_id)
    return dot


def plot_model(model, to_file=None, display_image=True, **kwargs):
    """
    Extend tf.keras.utils.plot_model

    Parameters:
        model: A Keras model instance
        to_file: File name of the plot image
        display_image:
    """
    dot = model_to_dot(model, **kwargs)

    if to_file is not None:
        from tensorflow.python.keras.utils.io_utils import path_to_string
        import os
        to_file = path_to_string(to_file)
        _, extension = os.path.splitext(to_file)
        extension = extension[1:]
        dot.write(to_file, format=extension)

    if display_image:
        try:
            from IPython import display
            if to_file is None or extension == 'svg':
                return display.SVG(dot.create(prog='dot', format='svg'))
            else:
                return display.Image(filename=to_file)
        except ImportError:
            pass


if __name__ == "__main__":
    from tensorflow.keras.applications import vgg16

    model = vgg16.VGG16(include_top=True)
    plot_model(model)
