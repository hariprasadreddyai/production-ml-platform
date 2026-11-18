"""Tests for drift detection (PSI, KS test, label distribution)."""

import numpy as np

from monitoring.drift_detector import (
    DriftDetector,
    ks_test,
    label_distribution_drift,
    population_stability_index,
)


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 1000)
    assert population_stability_index(sample, sample) < 0.01


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    shifted = rng.normal(2, 1, 1000)
    assert population_stability_index(reference, shifted) > 0.25


def test_ks_test_identical_samples():
    rng = np.random.default_rng(1)
    sample = rng.normal(0, 1, 500)
    result = ks_test(sample, sample)
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_ks_test_detects_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 500)
    shifted = rng.normal(3, 1, 500)
    result = ks_test(reference, shifted)
    assert result["statistic"] > 0.5
    assert result["p_value"] < 0.01


def test_label_drift_detects_shift():
    reference = ["positive"] * 80 + ["negative"] * 10 + ["neutral"] * 10
    current = ["positive"] * 10 + ["negative"] * 80 + ["neutral"] * 10
    result = label_distribution_drift(reference, current)
    assert result["drifted"] is True
    assert result["max_shift"] > 0.2


def test_label_drift_stable():
    reference = ["positive"] * 50 + ["negative"] * 30 + ["neutral"] * 20
    current = ["positive"] * 52 + ["negative"] * 28 + ["neutral"] * 20
    result = label_distribution_drift(reference, current)
    assert result["drifted"] is False


def test_drift_detector_combined_check():
    rng = np.random.default_rng(2)
    reference_features = {"text_length": rng.normal(50, 10, 500)}
    reference_labels = ["positive"] * 300 + ["negative"] * 200
    detector = DriftDetector(reference_features, reference_labels)

    stable = detector.check(
        {"text_length": rng.normal(50, 10, 500)},
        ["positive"] * 310 + ["negative"] * 190,
    )
    assert stable["drifted"] is False

    drifted = detector.check(
        {"text_length": rng.normal(120, 10, 500)},
        ["positive"] * 50 + ["negative"] * 450,
    )
    assert drifted["drifted"] is True
    assert drifted["features"]["features"]["text_length"]["drifted"] is True
