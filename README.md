# tfxtend

TensorFlow Extension Tools (**tfxtend**)

This is the tools of TensorFlow, Scikit-Learn, Hyperdash, and Plotly for me.

## Keras Callbacks

Callbacks of metrics (such as confusion matrix, f-measure and softmax values)

```python
from tfxtend.keras.callbacks import metrics, hyperdash

# Log confusion matrix of test data
cm_callback = metrics.ConfusionMatrixLogger(model, x_test, y_test, label_list=label_list, period=10, filepath="./confusion_matrix.csv")

# Log f-measure of test data
fm_callback = metrics.FMeasureLogger(model, x_test, y_test, label_list=label_list, period=10, filepath="./f-measure.csv")

# Log softmax values of test data
sm_callback = metrics.SoftmaxLogger(model, x_test, y_test, label_list=label_list, period=10, filepath="./softmax_values.csv")


# Training
stack = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_data=(x_test, y_test),
                  callbacks=[cm_callback, fm_callback, sm_callback])    
```

Callback for Hyperdash
```python
from hyperdash import Experiment
from tfxtend.keras.hyperdash import Hyperdash


exp = Experiment(monitor_name)
hd_callback = Hyperdash(["val_loss", "loss", "val_accuracy", "accuracy"], exp)

exp.param("epochs", epochs)
exp.param("batch size", batch)

# Training
stack = model.fit(x_train, y_train, batch_size=batch, epochs=epochs,
                  validation_data=(x_test, y_test), verbose=1,
                  callbacks=[hd_callbacks])
exp.end()
```

## Plot Keras Model

```python
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
```

A running example is [this Google Colab notebook](https://colab.research.google.com/drive/1nPVc1BXSwCqwhDtlULyYMteGHEZNNTpQ?usp=sharing).



## API reference

[tfxtend API reference](docs/reference.md)

