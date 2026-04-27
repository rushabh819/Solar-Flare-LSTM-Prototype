from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def tss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr - fpr


def hss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denom == 0:
        return 0.0
    return 2 * (tp * tn - fp * fn) / denom


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "tss": float(tss_score(y_true, y_pred)),
        "hss": float(hss_score(y_true, y_pred)),
    }
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    metrics.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return metrics


def optimize_threshold_for_tss(y_true: np.ndarray, y_prob: np.ndarray):
    candidates = np.linspace(0.05, 0.95, 19)
    scored = [evaluate_binary(y_true, y_prob, t) for t in candidates]
    return max(scored, key=lambda x: x["tss"])
