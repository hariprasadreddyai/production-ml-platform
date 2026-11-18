"""Drift detection: PSI and two-sample Kolmogorov-Smirnov tests.

Data drift compares live input distributions against a stored reference
(PSI for binned features, KS for continuous values). Model drift compares
the distribution of predicted labels. Both are pure numpy, no extra deps.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two samples. <0.1 stable, 0.1-0.25 moderate, >0.25 drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    expected_perc = np.maximum(expected_counts / max(len(expected), 1), 1e-4)
    actual_perc = np.maximum(actual_counts / max(len(actual), 1), 1e-4)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def ks_test(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test implemented with numpy."""
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, combined, side="right") / len(a)
    cdf_b = np.searchsorted(b, combined, side="right") / len(b)
    statistic = float(np.max(np.abs(cdf_a - cdf_b)))
    n = len(a) * len(b) / (len(a) + len(b))
    approx_p = float(np.exp(-2 * n * statistic**2)) if statistic > 0 else 1.0
    return {"statistic": round(statistic, 4), "p_value": round(approx_p, 4)}


def label_distribution_drift(reference_labels: list[str],
                             current_labels: list[str]) -> dict:
    """Model drift: compare predicted label distributions."""
    labels = sorted(set(reference_labels) | set(current_labels))
    ref_counts = {label: reference_labels.count(label) for label in labels}
    cur_counts = {label: current_labels.count(label) for label in labels}
    total_ref = max(len(reference_labels), 1)
    total_cur = max(len(current_labels), 1)
    shifts = {
        label: round(cur_counts[label] / total_cur - ref_counts[label] / total_ref, 4)
        for label in labels
    }
    max_shift = max((abs(v) for v in shifts.values()), default=0.0)
    return {
        "label_shifts": shifts,
        "max_shift": max_shift,
        "drifted": max_shift > 0.2,
    }


class DriftDetector:
    """Holds a reference snapshot and checks live batches against it."""

    def __init__(self, reference_features: dict[str, np.ndarray],
                 reference_labels: list[str],
                 psi_threshold: float = 0.25,
                 ks_p_threshold: float = 0.01):
        self.reference_features = reference_features
        self.reference_labels = reference_labels
        self.psi_threshold = psi_threshold
        self.ks_p_threshold = ks_p_threshold

    def check_features(self, current_features: dict[str, np.ndarray]) -> dict:
        results = {}
        drifted_any = False
        for name, current in current_features.items():
            reference = self.reference_features.get(name)
            if reference is None:
                continue
            psi = population_stability_index(reference, current)
            ks = ks_test(reference, current)
            drifted = psi > self.psi_threshold or ks["p_value"] < self.ks_p_threshold
            drifted_any = drifted_any or drifted
            results[name] = {"psi": round(psi, 4), "ks": ks, "drifted": drifted}
        return {"features": results, "drifted": drifted_any}

    def check_labels(self, current_labels: list[str]) -> dict:
        return label_distribution_drift(self.reference_labels, current_labels)

    def check(self, current_features: dict[str, np.ndarray],
              current_labels: list[str]) -> dict:
        feature_result = self.check_features(current_features)
        label_result = self.check_labels(current_labels)
        return {
            "features": feature_result,
            "labels": label_result,
            "drifted": feature_result["drifted"] or label_result["drifted"],
        }
