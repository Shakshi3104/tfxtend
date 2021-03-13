import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from ... import metrics


# Confusion Matrixを記録する
class ConfusionMatrixLogger(Callback):
    def __init__(self, model, x_test, y_test, label_list, period=10, filepath=None, filedir=None):
        """
        ConfusionMatrixLogger
            model: Model of tensorflow.keras
            x_test: np.ndarray
            y_test: np.ndarray, not one-hot vector
            label_list: list of target names, optional
            period: span of outputting confusion matrix, default `10`
            filepath: filepath of confusion matrix, ignored filedir is not None, optional
            filedir: filepath of directory of confusion matrix, optional
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.label_list = label_list
        self.period = period
        self.filepath = filepath
        self.filedir = filedir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            predict = self.model.predict(self.x_test)
            predict = np.argmax(predict, axis=1)
            cf = metrics.confusion_error_matrix(predict, self.y_test, target_names=self.label_list)
            if self.filedir is not None:
                cf.to_csv(self.filedir + "confusion_matrix_{}.csv".format(epoch + 1))
            elif self.filepath is not None:
                cf.to_csv(self.filepath)

            print("Confusion Matrix")
            print(cf)


# ラベルごとのF値を記録する
class FMeasureLogger(Callback):
    def __init__(self, model, x_test, y_test, label_list, period=10, filepath=None, filedir=None):
        """
        FMeasureLogger
            model: Model of tensorflow.keras
            x_test: np.ndarray
            y_test: np.ndarray, not one-hot vector
            label_list: list of target names
            period: span of outputting F-measure, default `10`
            filepath: filepath of F-measure (F1 score), ignored filedir is not None, optional
            filedir: filepath of directory of F-measure (F1 score), optional
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.label_list = label_list
        self.period = period
        self.filepath = filepath
        self.filedir = filedir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            predict = self.model.predict(self.x_test)
            predict = np.argmax(predict, axis=1)

            df = metrics.f_measure(self.y_test, predict, self.label_list, output_dict=False)

            if self.filedir is not None:
                df.to_csv(self.filedir + "f-measure_{}.csv".format(epoch + 1))
            elif self.filepath is not None:
                df.to_csv(self.filepath, index=False)

            print("F-measure")
            print(df)


class SoftmaxLogger(Callback):
    def __init__(self, model, x_test, y_test, label_list, period=10, filepath=None, filedir=None):
        """
        SoftmaxLogger
            model: Model of tensorflow.keras
            x_test: np.ndarray
            y_test: np.ndarray, not one-hot vector
            label_list: list of target names
            period: span of outputting F-measure, default `10`
            filepath: filepath of Softmax value, ignored filedir is not None, optional
            filedir: filepath of directory of softmax value, optional
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.label_list = label_list
        self.period = period
        self.filepath = filepath
        self.filedir = filedir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            predict = self.model.predict(self.x_test)
            df = pd.DataFrame(predict, columns=self.label_list)
            predict = np.argmax(predict, axis=1)
            predict = predict.tolist()

            if self.label_list is not None:
                df["Predict"] = [self.label_list[i] for i in predict]
                df["Correct"] = [self.label_list[i] for i in self.y_test]

            if self.filedir is not None:
                df.to_csv(self.filedir + "softmax_values_{}.csv".format(epoch + 1))

            elif self.filepath is not None:
                df.to_csv(self.filepath, index=False)
