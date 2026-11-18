"""Classification and latency metrics for model evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0)), 4,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, labels=labels,
                               average="macro", zero_division=0)), 4,
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, labels=labels,
                           average="macro", zero_division=0)), 4,
        ),
    }


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    values = np.asarray(latencies_ms, dtype=float)
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "mean_ms": round(float(values.mean()), 2),
    }
