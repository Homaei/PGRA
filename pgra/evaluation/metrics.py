"""
Evaluation metrics for the PGRA framework.

F1/Precision/Recall are computed binary-style on the actually-labelled
portion of the test set. ASR is computed on D_target (held-out
backdoor samples): an attack is "successful" if the model predicts
NORMAL for a sample that was crafted to look spoofed.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_f1_precision_recall(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return float(f1), float(precision), float(recall)


def compute_asr(y_pred_on_target):
    """
    y_pred_on_target: predictions (0/1) on each sample of D_target.
    All D_target samples carry a spoofed H_attack (ground-truth class 1).
    Attack success := predicted as normal (0).
    """
    y = np.asarray(y_pred_on_target, dtype=int)
    if y.size == 0:
        return 0.0
    return float((y == 0).mean())
