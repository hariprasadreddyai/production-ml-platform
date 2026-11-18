"""Tests for the evaluation engine and CI gate."""

import json
from pathlib import Path

from evaluation.ci_gate import run_gate
from evaluation.evaluator import Evaluator
from evaluation.golden_test_runner import (
    GoldenTestCase,
    load_test_cases,
    run_golden_tests,
)
from evaluation.metrics import classification_metrics, latency_stats
from evaluation.regression_detector import compare_runs, detect_regression
from serving.inference import lexicon_predict

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data/golden_tests/test_cases.json"


def test_load_test_cases():
    cases = load_test_cases(GOLDEN_PATH)
    assert len(cases) >= 20
    assert cases[0].id == "TC001"
    assert cases[0].expected_label == "positive"
    assert cases[1].expected_label == "negative"
    assert {c.expected_label for c in cases} == {"positive", "negative", "neutral"}


def test_golden_run_with_stub_predictor_is_accurate():
    cases = load_test_cases(GOLDEN_PATH)
    run = run_golden_tests(cases, lexicon_predict)
    assert run.accuracy >= 0.95
    assert all(r.confidence_met for r in run.results)


def test_classification_metrics():
    metrics = classification_metrics(
        ["positive", "negative", "neutral", "positive"],
        ["positive", "negative", "neutral", "negative"],
    )
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["f1"] <= 1.0


def test_latency_stats():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["p50_ms"] == 30.0
    assert stats["mean_ms"] == 30.0
    assert stats["p99_ms"] > stats["p50_ms"]


def test_latency_stats_empty():
    assert latency_stats([]) == {
        "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0,
    }


def test_regression_detector_flags_drop():
    baseline = {"accuracy": 0.95, "f1": 0.93, "p99_ms": 20.0}
    current = {"accuracy": 0.90, "f1": 0.88, "p99_ms": 25.0}
    result = detect_regression(baseline, current, max_regression_drop=0.02)
    assert result["regressed"] is True
    assert result["accuracy_drop"] == 0.05


def test_regression_detector_passes_small_drop():
    baseline = {"accuracy": 0.95, "f1": 0.93, "p99_ms": 20.0}
    current = {"accuracy": 0.94, "f1": 0.92, "p99_ms": 21.0}
    result = detect_regression(baseline, current, max_regression_drop=0.02)
    assert result["regressed"] is False


def test_compare_runs_finds_flipped_cases():
    cases = [
        GoldenTestCase(id="TC1", input="I love it", expected_label="positive"),
        GoldenTestCase(id="TC2", input="This is awful", expected_label="negative"),
    ]
    baseline = run_golden_tests(cases, lexicon_predict)

    def bad_predict(text):
        return "negative", 0.9

    current = run_golden_tests(cases, bad_predict)
    result = compare_runs(baseline, current)
    assert result["regressed_cases"] == ["TC1"]
    assert result["regression_count"] == 1


def test_evaluator_thresholds_pass(tmp_path):
    evaluator = Evaluator(str(GOLDEN_PATH))
    result = evaluator.evaluate(lexicon_predict)
    gate = evaluator.check_thresholds(result["summary"])
    assert gate["passed"] is True
    assert result["summary"]["accuracy"] >= 0.85


def test_ci_gate_passes_end_to_end(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "evaluation:\n"
        f"  golden_test_path: '{GOLDEN_PATH}'\n"
        "thresholds:\n"
        "  min_accuracy: 0.85\n"
        "  min_f1: 0.80\n"
        "  max_latency_p99_ms: 50\n"
        "  max_regression_drop: 0.02\n"
        "ci_gate:\n"
        "  block_on_failure: true\n"
        "  report_format: json\n"
    )
    report = tmp_path / "quality.json"
    result = run_gate(str(config_path), report_path=str(report))
    assert result["gate"]["passed"] is True
    assert report.exists()
    saved = json.loads(report.read_text())
    assert saved["summary"]["accuracy"] >= 0.85
    assert Path(str(report).replace(".json", ".txt")).exists()


def test_ci_gate_fails_when_thresholds_unreachable(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "evaluation:\n"
        f"  golden_test_path: '{GOLDEN_PATH}'\n"
        "thresholds:\n"
        "  min_accuracy: 1.01\n"
        "  min_f1: 0.80\n"
        "  max_latency_p99_ms: 50\n"
        "  max_regression_drop: 0.02\n"
        "ci_gate:\n"
        "  block_on_failure: true\n"
        "  report_format: json\n"
    )
    result = run_gate(str(config_path))
    assert result["gate"]["passed"] is False
    assert result["gate"]["checks"]["accuracy"] is False
