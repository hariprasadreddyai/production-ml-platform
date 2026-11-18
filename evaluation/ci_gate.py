"""CI quality gate: run golden tests and block deployment on failure.

Reads thresholds from the evaluation config, runs the golden test suite
against the predictor (the stub classifier by default, or a saved model
with --model), writes a JSON report plus a human-readable summary, and
exits non-zero when any check fails so CI blocks the deploy.

Usage:
    python -m evaluation.ci_gate --config configs/evaluation_config.yaml
    python -m evaluation.ci_gate --config configs/evaluation_config.yaml \
        --model models/latest
    python -m evaluation.ci_gate --config configs/evaluation_config.yaml \
        --report reports/quality.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evaluation.evaluator import Evaluator
from evaluation.regression_detector import detect_regression


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_predictor(model_path: str | None):
    if model_path is None:
        from serving.inference import lexicon_predict

        return lexicon_predict
    from serving.inference import HFModel

    model = HFModel(model_path)
    return model.predict


def render_human_report(result: dict) -> str:
    summary = result["summary"]
    gate = result["gate"]
    lines = [
        "MLServe Quality Report",
        "======================",
        f"Overall: {'PASS' if gate['passed'] else 'FAIL'}",
        "",
        "Metrics:",
        f"  accuracy:  {summary['accuracy']:.4f}",
        f"  precision: {summary['precision']:.4f}",
        f"  recall:    {summary['recall']:.4f}",
        f"  f1:        {summary['f1']:.4f}",
        f"  p50 latency: {summary['p50_ms']:.2f} ms",
        f"  p95 latency: {summary['p95_ms']:.2f} ms",
        f"  p99 latency: {summary['p99_ms']:.2f} ms",
        f"  cases: {summary['total_cases']}",
        "",
        "Threshold checks:",
    ]
    for name, ok in gate["checks"].items():
        lines.append(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if result.get("regression"):
        reg = result["regression"]
        lines += [
            "",
            "Regression vs baseline:",
            f"  regressed: {reg['regressed']}",
            f"  {reg['details']}",
        ]
    failed = [r for r in result["run"]["results"] if not r["passed"]]
    if failed:
        lines += ["", "Failed cases:"]
        for r in failed:
            lines.append(
                f"  {r['id']}: expected {r['expected_label']}, "
                f"got {r['predicted_label']} ({r['input'][:60]})"
            )
    return "\n".join(lines)


def run_gate(config_path: str, model_path: str | None = None,
             report_path: str | None = None,
             baseline_path: str | None = None) -> dict:
    config = load_config(config_path)
    thresholds = config.get("thresholds", {})
    golden_path = config["evaluation"]["golden_test_path"]
    ci_gate = config.get("ci_gate", {})

    evaluator = Evaluator(golden_path, thresholds)
    predict = build_predictor(model_path)
    result = evaluator.evaluate(predict)
    gate = evaluator.check_thresholds(result["summary"])
    result["gate"] = gate

    if baseline_path:
        with open(baseline_path) as f:
            baseline = json.load(f)["summary"]
        result["regression"] = detect_regression(
            baseline, result["summary"],
            max_regression_drop=thresholds.get("max_regression_drop", 0.02),
        )
        if result["regression"]["regressed"]:
            gate["passed"] = False
            gate["checks"]["regression"] = False

    result["report_format"] = ci_gate.get("report_format", "json")
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(render_human_report(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CI quality gate")
    parser.add_argument("--config", default="configs/evaluation_config.yaml")
    parser.add_argument("--model", default=None,
                        help="Path to a saved model directory (default: stub)")
    parser.add_argument("--report", default="reports/quality.json")
    parser.add_argument("--baseline", default=None,
                        help="Previous quality report JSON for regression checks")
    args = parser.parse_args()

    result = run_gate(args.config, model_path=args.model,
                      report_path=args.report, baseline_path=args.baseline)
    print(render_human_report(result))
    if result["gate"]["passed"]:
        print("\nQuality gate PASSED.")
        return 0
    print("\nQuality gate FAILED. Deployment blocked.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
