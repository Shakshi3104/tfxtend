import time
import platform

from sklearn.base import BaseEstimator
import sklearn


class EstimatorPerformance(BaseEstimator):
    def __init__(self, clf: BaseEstimator, filepath=None):
        self.clf = clf
        self.filepath = filepath

    def fit(self, X, y):
        train_start_time = time.time()
        self.clf.fit(X, y)
        train_elapsed_time = time.time() - train_start_time

        # system information
        system_info = platform.system()
        system_name = "system: {}".format(system_info)
        platform_info = "platform: {}".format(platform.platform())
        sk_version = "sklearn.__version__: {}".format(sklearn.__version__)

        # only mac
        if system_info == "Darwin":
            system_name = "{} ({})".format(system_name, platform.machine())

        # result
        train_time = "training: {}s".format(int(train_elapsed_time) if train_elapsed_time >= 1 else round(train_elapsed_time, 3))

        # print result
        result = [system_name, platform_info, sk_version, train_time]

        for result_ in result:
            print(result_)

        if self.filepath is not None:
            with open(self.filepath, "w") as f:
                for result_ in result:
                    print(result_, file=f)

        return self

    def predict(self, X):
        predict_start_time = time.time()
        predict = self.clf.predict(X)
        predict_elapsed_time = time.time() - predict_start_time

        print("predicting: {}s".format(int(predict_elapsed_time) if predict_elapsed_time >= 1 else round(predict_elapsed_time, 3)))
        return predict


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()

    clf = RandomForestClassifier()
    bench = EstimatorPerformance(clf)

    bench.fit(iris.data, iris.target)
    bench.predict(iris.data)
