"""Golden test dataset runner.

Runs a JSON file of labeled test cases against any predictor callable and
records per-case results: predicted label, confidence, whether the label
matched, whether the confidence floor was met, and latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenTestCase:
    id: str
    input: str
    expected_label: str
    min_confidence: float = 0.5
    category: str = "general"


@dataclass
class CaseResult:
    id: str
    input: str
    expected_label: str
    predicted_label: str
    confidence: float
    passed: bool
    confidence_met: bool
    latency_ms: float
    category: str


@dataclass
class GoldenRun:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def latencies_ms(self) -> list[float]:
        return [r.latency_ms for r in self.results]

    @property
    def y_true(self) -> list[str]:
        return [r.expected_label for r in self.results]

    @property
    def y_pred(self) -> list[str]:
        return [r.predicted_label for r in self.results]

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "total": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "failed": sum(not r.passed for r in self.results),
            "results": [
                {
                    "id": r.id,
                    "input": r.input,
                    "expected_label": r.expected_label,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "passed": r.passed,
                    "confidence_met": r.confidence_met,
                    "latency_ms": round(r.latency_ms, 2),
                    "category": r.category,
                }
                for r in self.results
            ],
        }


def load_test_cases(path: str | Path) -> list[GoldenTestCase]:
    with open(path) as f:
        raw = json.load(f)
    return [GoldenTestCase(**case) for case in raw]


def run_golden_tests(cases: list[GoldenTestCase], predict) -> GoldenRun:
    """Run cases against predict(text) -> (label, confidence)."""
    run = GoldenRun()
    for case in cases:
        start = time.perf_counter()
        label, confidence = predict(case.input)
        latency_ms = (time.perf_counter() - start) * 1000
        run.results.append(
            CaseResult(
                id=case.id,
                input=case.input,
                expected_label=case.expected_label,
                predicted_label=label,
                confidence=confidence,
                passed=label == case.expected_label,
                confidence_met=confidence >= case.min_confidence,
                latency_ms=latency_ms,
                category=case.category,
            )
        )
    return run
