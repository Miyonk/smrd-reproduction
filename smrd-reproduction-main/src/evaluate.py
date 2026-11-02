"""
Evaluation metrics and statistical utilities.

Specification: Section 7 of the technical design (Equations 15-22, Wilson
confidence intervals, McNemar and bootstrap tests).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> List[List[int]]:
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth)][int(prediction)] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> Dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "mcc": 0.0,
            "npv": 0.0,
            "per_class": [],
            "confusion_matrix": [[0] * n_classes for _ in range(n_classes)],
        }

    matrix = confusion_matrix(y_true, y_pred, n_classes)
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(n_classes))
    per_class = []
    for c in range(n_classes):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n_classes) if r != c)
        fn = sum(matrix[c][k] for k in range(n_classes) if k != c)
        tn = total - tp - fp - fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "class": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = float(np.mean([m["precision"] for m in per_class]))
    macro_recall = float(np.mean([m["recall"] for m in per_class]))
    macro_f1 = float(np.mean([m["f1"] for m in per_class]))

    # Binary metrics use class 1 as positive, matching Arm B/D label mapping.
    if n_classes == 2:
        m = per_class[1]
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        precision = m["precision"]
        recall = m["recall"]
        f1 = m["f1"]
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = _safe_div(tp * tn - fp * fn, mcc_denom)
        npv = _safe_div(tn, tn + fn)
    else:
        precision = macro_precision
        recall = macro_recall
        f1 = macro_f1
        fpr = float(np.mean([_safe_div(m["fp"], m["fp"] + m["tn"]) for m in per_class]))
        fnr = float(np.mean([_safe_div(m["fn"], m["fn"] + m["tp"]) for m in per_class]))
        mcc = 0.0
        npv = float(np.mean([_safe_div(m["tn"], m["tn"] + m["fn"]) for m in per_class]))

    return {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "npv": npv,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    delta = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denom
    )
    return (max(0.0, centre - delta), min(1.0, centre + delta))


def mcnemar_test(
    y_true: Sequence[int], predictions_a: Sequence[int], predictions_b: Sequence[int]
) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary correctness."""
    if not (len(y_true) == len(predictions_a) == len(predictions_b)):
        raise ValueError("all inputs must have equal length")
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    for truth, pred_a, pred_b in zip(y_true, predictions_a, predictions_b):
        correct_a = int(pred_a) == int(truth)
        correct_b = int(pred_b) == int(truth)
        if correct_a and not correct_b:
            b += 1
        elif not correct_a and correct_b:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    # Exact binomial two-sided p-value under p=0.5.
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    statistic = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value}


def bootstrap_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have equal length")
    rng = np.random.default_rng(seed)
    n = len(truth)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        values[i] = float(np.mean(pred[indices] == truth[indices]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n_resamples > 1 else 0.0,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def report_c_collapse(predictions: Sequence[int]) -> List[int]:
    """Collapse 4-class Arm A predictions to binary rumour/non-rumour."""
    return [0 if int(p) == 0 else 1 for p in predictions]


def summarize_binary(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict:
    metrics = classification_metrics(y_true, y_pred, 2)
    correct = sum(int(a) == int(b) for a, b in zip(y_true, y_pred))
    ci_low, ci_high = wilson_ci(correct, len(y_true))
    metrics["wilson_ci_95"] = [ci_low, ci_high]
    return metrics
