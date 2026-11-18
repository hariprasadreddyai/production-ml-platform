"""Regression detection: compare a new model against a baseline.

A regression is flagged when accuracy or F1 drops by more than the allowed
amount, or when p99 latency grows past the allowed ceiling. The comparison
uses the same golden test run for both models so results are apples to
apples.
"""

from __future__ import annotations

from evaluation.golden_test_runner import GoldenRun


def detect_regression(baseline: dict, current: dict,
                      max_regression_drop: float = 0.02) -> dict:
    """Compare aggregate metrics and report whether a regression occurred.

    baseline and current are dicts with accuracy, f1, and p99_ms keys.
    """
    accuracy_drop = baseline["accuracy"] - current["accuracy"]
    f1_drop = baseline["f1"] - current["f1"]
    latency_growth = current["p99_ms"] - baseline["p99_ms"]
    regressed = (
        accuracy_drop > max_regression_drop or f1_drop > max_regression_drop
    )
    return {
        "regressed": regressed,
        "accuracy_drop": round(accuracy_drop, 4),
        "f1_drop": round(f1_drop, 4),
        "latency_growth_ms": round(latency_growth, 2),
        "details": (
            f"accuracy dropped {accuracy_drop:.4f}, "
            f"F1 dropped {f1_drop:.4f}, p99 grew {latency_growth:.2f}ms"
        ),
    }


def compare_runs(baseline_run: GoldenRun, current_run: GoldenRun,
                 max_regression_drop: float = 0.02) -> dict:
    """Per-case comparison: which test cases flipped from pass to fail."""
    baseline_by_id = {r.id: r for r in baseline_run.results}
    regressions = []
    improvements = []
    for result in current_run.results:
        previous = baseline_by_id.get(result.id)
        if previous is None:
            continue
        if previous.passed and not result.passed:
            regressions.append(result.id)
        elif not previous.passed and result.passed:
            improvements.append(result.id)
    return {
        "regressed_cases": regressions,
        "improved_cases": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "regressed": len(regressions) > 0 and (
            baseline_run.accuracy - current_run.accuracy
            > max_regression_drop
        ),
    }
