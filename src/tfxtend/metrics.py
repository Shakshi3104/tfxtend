import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# Wrapping sklearn's confusion matrix
def confusion_error_matrix(y_row, y_col, target_names=None, normalize=False):
    """
    Wrapper confusion_matrix of sklearn

    Parameters
        y_row & y_col: if y_row is y_pred and y_col is y_true,
                        Confusion Matrix is `Pre. / Cor.`
                       if y_row is y_true and y_col is y_pred,
                        Confusion Matrix is `Cor. / Pre.`
        target_names: [string], List of target(label) name
        normalize: bool: if normalize is True, confusion matrix is normalized, default False
    Returns
        conf_max: pd.DataFrame: confusion matrix
    """
    conf_mat = confusion_matrix(y_row, y_col)
    if normalize:
        # 正規化処理
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)[:, np.newaxis]

    if target_names is not None:
        conf_mat = pd.DataFrame(conf_mat, columns=target_names, index=target_names)
    else:
        conf_mat = pd.DataFrame(conf_mat)

    return conf_mat


# F-measure
def f_measure(y_true, y_pred, target_names, output_dict=False):
    """
    Calculate f-measure (f1-score)

    Parameters
        y_true: 1d-array like: ground truth (correct) target values
        y_pred: 1d-array like: estimated targets as returned by a classifier
        target_names: list: target class labels, display names matching the labels (same order)
        output_dict: bool: if True, return output as dict. else return output as pd.DataFrame
    Returns
        f1_score: dict | pd.DataFrame: f-measure, if output_dict is True, return as dict
    """
    cr = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    f1_score = []

    # ラベルごとのF値を取得
    for label in target_names:
        r_label = cr[label]
        f1_ = r_label["f1-score"]
        f1_score += [f1_]

    if output_dict:
        return dict(zip(target_names, f1_score))
    else:
        return pd.DataFrame({"F-measure": f1_score, "Label": target_names})


# Confusion Matrix to Accuracy
def confusion_matrix_to_accuracy(conf_mat):
    """
    Convert confusion matrix to accuracy

    Parameters
        conf_mat: np.ndarray | pd.DataFrame: confusion matrix, length of columns and that of rows must be same.

    Returns
        accuracy: float64: accuracy
    """
    if type(conf_mat) is pd.DataFrame:
        conf_mat = conf_mat.values

    accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    return accuracy
