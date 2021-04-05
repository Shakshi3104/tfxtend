# keras
## callbacks
### tfxtend.keras.callbacks.metrics.ConfusionMatrixLogger
```python
tfxtend.keras.callbacks.metrics.ConfusionMatrixLogger(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)
```
Logger of confusion matrix of validation data on training

#### `__init__(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)`
##### Parameters
- model: `tensorflow.keras.models.Model`: `Model` object on training
- x_test: `np.ndarray`: input data for validation
- y_test: `np.ndarray`: target data for validation, **not one-hot vector**
- label_list: `list`: target class labels
- period: `int`: times of outputting the confusion matrix, default 10
- filename: `str`: the absolute path to the file to be saved, ignored if filedir is not `None`, default `None`
- filedir: `str`: the absolute path to the directory to be saved, save confusion matrix per period, default `None`

### tfxtend.keras.callbacks.metrics.FMeasureLogger
```python
tfxtend.keras.callbacks.metrics.FMeasureLogger(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)
```
Logger of F-measure of validation data on training

#### `__init__(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)`
##### Parameters
- model: `tensorflow.keras.models.Model`: `Model` object on training
- x_test: `np.ndarray`: input data for validation
- y_test: `np.ndarray`: target data for validation, **not one-hot vector**
- label_list: `list`: target class labels
- period: `int`: times of outputting the confusion matrix, default 10
- filename: `str`: the absolute path to the file to be saved, ignored if filedir is not `None`, default `None`
- filedir: `str`: the absolute path to the directory to be saved, save F-measure per period, default `None`

### tfxtend.keras.callbacks.metrics.SoftmaxLogger
```python
tfxtend.keras.callbacks.metrics.SoftmaxLogger(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)
```
Logger of Softmax value of each validation data on training

#### `__init__(model, x_test, y_test, label_list, period=10, filepath=None, filedir=None)`
##### Parameters
- model: `tensorflow.keras.models.Model`: `Model` object on training
- x_test: `np.ndarray`: input data for validation
- y_test: `np.ndarray`: target data for validation, **not one-hot vector**
- label_list: `list`: target class labels, default `None`
- period: `int`: times of outputting the confusion matrix, default 10
- filename: `str`: the absolute path to the file to be saved, ignored if filedir is not `None`, default `None`
- filedir: `str`: the absoluet path to the directory to be saved, save softmax values per period, default `None`

## Keras callbacks for Hyperdash
### tfxtend.keras.hyperdash.Hyperdash
```python
tfxtend.keras.hyperdash.Hyperdash(entries, exp)
```
callback for Hyperfash in Keras

[Hyperdash](https://hyperdash.io) is a machine learning monitoring library capable of running alongside Tensorflow, Scikit-Learn, and other modeling libraries.

#### `__init__(entries, exp)`
##### Parameters
- entries: `list`: list of metrics names to monitor; `["val_loss", "loss", "val_accuracy", "accuracy"]`
- exp: `hyperdash.Experiment`: `Experiment` object

## Plot tools
### tfxtend.keras.plotting.plot_learning_curve
```python
tfxtend.keras.plotting.plot_learning_curve(filename, epochs, stack=None, history=None, dark=False)
```
plot learning curve of `tensorflow.keras`
##### Parameters
- filename: `str`: the absolute path to the file to be saved
- epochs: `int`: the number of training iterations
- stack: `tensorflow.keras.History`: `History` object; returns of `tensorflow.keras.models.Model.fit`, default `None`
- history: `tensorflow.keras.History.history`: `History.history` object, default `None`
- dark: `bool`: if dark is `True`, learning curve is plotted in dark mode, default `False`

### tfxtend.keras.plotting.plot_feature_map
```python
tfxtend.keras.plotting.plot_feature_map(layer_outputs, y, label_list, filename=None, fig_len=10, umap=False, dark=False)
```
plot feature map by using T-SNE or UMAP
##### Parameters
- layer_outputs: output of layer with 2D shape; such as `(1797, 64)`
- y: class label
- filename: `str`: the absolute path to the file to be saved, default `None`
- fig_len: `int`: the length of figure, default 18
- umap: `bool`: if umap is `True`, use UMAP instead of T-SNE, default `False`
- dark: `bool`: if dark is `True`, feature map is plotted in dark mode, default `False`

##### Returns
- x_cmprss: `pd.DataFrame`: 2D compressed output of layer

### tfxtend.keras.plotting.model_to_dot
```python
tfxtend.keras.plotting.model_to_dot(model, show_shapes=False, show_layer_names=False, show_params=False, show_configs=False, splines=True, concentrate=True, fillcolor='white', bgcolor='white', fgcolor='black', horizontal=False, dpi=None, config_list=None, colormap=None)
```

tensorflow.keras.utils.model_to_dot extension. Original implementation is https://qiita.com/T-STAR/items/f3adf908a7be8e5731ff

### tfxtend.keras.plotting.plot_model
```python
tfxtend.keras.plotting.plot_model(model, to_file=None, display_image=True, **kwargs)
```

tensorflow.keras.utils.plot_model extension. Original implementation is https://qiita.com/T-STAR/items/f3adf908a7be8e5731ff

##### Example

```python
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
```

A running example is [this Google Colab notebook](https://colab.research.google.com/drive/1nPVc1BXSwCqwhDtlULyYMteGHEZNNTpQ?usp=sharing).


# plotting
## Plotly
### tfxtend.plotting.plotly.SeabornColorPalette.to_plotly_rgb
```python
tfxtend.plotting.plotly.SeabornColorPalette.to_plotly_rgb(colorpalette, num_color)
```
converter `seaborn`'s color palette to `plotly` rgb format

##### Parameters
- colorpalette: `str`: the name of color palette of `seaborn`; the list of `seaborn`'s color palette is `tfxtend.plotting.plotly.SeabornColorPalette.palette_names`
- num_color: `int`: the number of color

##### Returns
- rgb: `list`: list of `plotly` rgb format

### tfxtend.plotting.plotly.plotly_boxplot

```python
tfxtend.plotting.plotly.plotly_boxplot(data, x, y, pallet='Pastel1', title=None, width=1200, height=700,
                                       showmeans=False, dark=False)
```

Boxplot using Plotly

##### Parameters
- data: `pd.DataFrame`: data to plot
- x: `str`: the column for x-axis
- y: `str`: the column for y-axis
- pallete: `str`: the name of color palette of `seaborn`; the list of `seaborn`'s color palette is `tfxtend.plotting.plotly.SeabornColorPalette.palette_names`, default `'Pastel1'`
- title: `str`: the title of plot, default `None`
- width: `int`: width of plot, default 1200
- height: `int`: height of plot, default 700
- showmeans: `bool`: if showmeans is `True`, plot average line in boxplot, default `False`
- dark: `bool`: if dark is `True`, boxplot is plotted in dark mode, default `False`

##### Returns
- fig: `plotly.graph_objects.Figure`: `Figure` object

### tfxtend.plotting.plotly.plotly_heatmap
```python
tfxtend.plotting.plotly.plotly_heatmap(data, cmap='Blues', side=500, dark=False)
```

Heatmap using Plotly

##### Parameters
- data: `pd.DataFrame`: data to plot
- cmap: `str`: the name of color palette of `seaborn`; the list of `seaborn`'s color palette is `tfxtend.plotting.plotly.SeabornColorPalette.palette_names`, default `'Blues'`
- side: `int`: the length of side of plot, default 500
- dark: `bool`: if dark is `True`, heatmap is plotted in dark mode, default `False`

#### Returns
- fig: `plotly.graph_objects.Figure`: `Figure` object

### tfxtend.plotting.plotly.plotly_multi_boxplot
```python
tfxtend.plotting.plotly.plotly_multi_boxplot(data, x, y, hue, pallet='Pastel1', title=None, width=1200, height=700, showmeans=False, dark=False)
```

Multi boxplot using Plotly

##### Parameters
- data: `pd.DataFrame`: data to plot
- x: `str`: the column for x-axis
- y: `str`: the column for y-axis
- hue: `str`: the column for multi
- pallete: `str`: the name of color palette of `seaborn`; the list of `seaborn`'s color palette is `tfxtend.plotting.plotly.SeabornColorPalette.palette_names`, default `'Pastel1'`
- title: `str`: the title of plot, default `None`
- width: `int`: width of plot, default 1200
- height: `int`: height of plot, default 700
- showmeans: `bool`: if showmeans is `True`, plot average line in boxplot, default `False`
- dark: `bool`: if dark is `True`, boxplot is plotted in dark mode, default `False`

##### Returns
- fig: `plotly.graph_objects.Figure`: `Figure` object

# Metrics
## Wrapper of Scikit-Learn
### tfxtend.metrics.confusion_error_matrix
```python
tfxtend.metrics.confusion_error_matrix(y_row, y_col, target_names, normalize=False)
```
confusion matrix, wrapping `sklearn.metrics.confusion_matrix`

##### Parameters
- y_row & y_col: *1d-array like*: if y_row is y_pred and y_col is y_true, confusion matrix is `Pre. / Cor.`. if y_row is y_true and y_col is y_pred, confusion matrix is `Cor. / Pre.`
- target_names: `list`: target class labels, display names matching the labels (same order)
- normalize: `bool`: if normalize is `True`, confusion matrix is normalized, default `False`

##### Returns
- conf_max: `pd.DataFrame`: confusion matrix

### tfxtend.metrics.f_measure
```python
tfxtend.metrics.f_measure(y_true, y_pred, target_names, output_dict=False)
```

F-measure, wrapping `sklearn.metrics.classification_report`

##### Parameters
- y_true: *1d-array like*: ground truth (correct) target values
- y_pred: *1d-array like*: estimated targets as returned by a classifier
- target_names: `list`: target class labels, display names matching the labels (same order)
- output_dict: `bool`: if True, return output as dict. else return output as pd.DataFrame

### tfxtend.metrics.confusion_matrix_to_accuracy
```python
tfxtend.metrics.confusion_matrix_to_accuracy(conf_mat)
```

Convert confusion matrix to accuracy

##### Parameters
- conf_mat: `np.ndarray` | `pd.DataFrame`: confusion matrix, length of columns and that of rows must be same.

##### Returns
- accuracy: `float64`: accuracy