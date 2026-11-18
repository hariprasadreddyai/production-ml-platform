"""Evaluator: run golden tests, compute metrics, check thresholds."""

from __future__ import annotations

from evaluation.golden_test_runner import GoldenRun, load_test_cases, run_golden_tests
from evaluation.metrics import classification_metrics, latency_stats


class Evaluator:
    def __init__(self, golden_test_path: str, thresholds: dict | None = None):
        self.golden_test_path = golden_test_path
        self.thresholds = thresholds or {
            "min_accuracy": 0.85,
            "min_f1": 0.80,
            "max_latency_p99_ms": 50,
            "max_regression_drop": 0.02,
        }

    def evaluate(self, predict) -> dict:
        cases = load_test_cases(self.golden_test_path)
        run: GoldenRun = run_golden_tests(cases, predict)
        cls_metrics = classification_metrics(run.y_true, run.y_pred)
        lat = latency_stats(run.latencies_ms)
        summary = {**cls_metrics, **lat}
        summary["total_cases"] = len(cases)
        summary["min_confidence_pass_rate"] = round(
            sum(r.confidence_met for r in run.results) / max(len(run.results), 1),
            4,
        )
        return {"summary": summary, "run": run.to_dict()}

    def check_thresholds(self, summary: dict) -> dict:
        checks = {
            "accuracy": summary["accuracy"] >= self.thresholds["min_accuracy"],
            "f1": summary["f1"] >= self.thresholds["min_f1"],
            "p99_latency": summary["p99_ms"] <= self.thresholds["max_latency_p99_ms"],
        }
        return {"passed": all(checks.values()), "checks": checks}
