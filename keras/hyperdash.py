from tensorflow.keras.callbacks import Callback


# Hyperdash用のKerasのCallback関数
class Hyperdash(Callback):
    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        print("epoch " + str(epoch + 1))

        for entry in self.entries:
            log = logs.get(entry)
            if log is not None:
                self.exp.metric(entry, log)


# Hyperdashの記録する値を登録する
def register_experiment(exp, dictionary):
    for name, value in dictionary.items():
        exp.param(name, value)