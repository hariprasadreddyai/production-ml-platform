"""Evaluation engine: golden tests, metrics, regression detection, CI gate."""

from evaluation.evaluator import Evaluator
from evaluation.golden_test_runner import load_test_cases, run_golden_tests

__all__ = ["Evaluator", "load_test_cases", "run_golden_tests"]
