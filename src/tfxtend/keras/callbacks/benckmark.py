import numpy as np
import pandas as pd
import time
import platform

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class PerformanceLogger(Callback):
    """Benchmarking training performance
        Logging time to train a model, time per epoch, and hardware information.
    """
    def __init__(self, filepath=None):
        self.train_start_time = None
        self.epoch_start_time = None

        self.epoch_elapsed_time = []
        self.train_elapsed_time = None

        self.filepath = filepath

    # training start
    # fit() start
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    # epoch start
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    # epoch end
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_elapsed_time.append(time.time() - self.epoch_start_time)

    # training end
    def on_train_end(self, logs=None):
        self.train_elapsed_time = time.time() - self.train_start_time

        # system information
        system_info = platform.system()
        system_name = "system: {}".format(system_info)
        platform_info = "platform: {}".format(platform.platform())
        tf_version = "tf.__version__: {}".format(tf.__version__)
        gpu_info = "tf.test.gpu_device_name(): {}".format(tf.test.gpu_device_name())

        # only mac
        if system_info == "Darwin":
            system_name = "{} ({})".format(system_name, platform.machine())

        # result
        train_time = "training: {}s ({} epochs)".format(int(self.train_elapsed_time), len(self.epoch_elapsed_time))
        epoch_time = "epoch: {}s".format(int(np.mean(self.epoch_elapsed_time)) if np.mean(self.epoch_elapsed_time) >= 1 else round(np.mean(self.epoch_elapsed_time), 3))

        # print result
        result = [system_name, platform_info, tf_version, gpu_info, train_time, epoch_time]

        for result_ in result:
            print(result_)

        if self.filepath is not None:
            with open(self.filepath, "w") as f:
                for result_ in result:
                    print(result_, file=f)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    callback = PerformanceLogger()

    model.fit(x_train, y_train, batch_size=128, epochs=10,
              callbacks=[callback])

