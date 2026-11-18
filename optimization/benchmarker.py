"""Latency and throughput benchmarking for ONNX models.

Measures p50, p95, and p99 latency plus requests per second over a
configurable number of runs, and can compare two models side by side in a
before/after report.

Usage:
    python -m optimization.benchmarker --model models/latest.onnx
    python -m optimization.benchmarker --model models/latest.int8.onnx --baseline models/latest.onnx
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import onnxruntime as ort


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def benchmark_model(model_path: str, runs: int = 100, warmup: int = 10,
                    seq_length: int = 128) -> dict:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_ids = np.ones((1, seq_length), dtype=np.int64)
    attention_mask = np.ones((1, seq_length), dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}

    for _ in range(warmup):
        session.run(None, feed)

    latencies: list[float] = []
    start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)
    total_seconds = time.perf_counter() - start

    return {
        "model": model_path,
        "runs": runs,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
        "throughput_rps": round(runs / total_seconds, 2),
    }


def compare_models(baseline: dict, optimized: dict) -> dict:
    """Build a before/after comparison report from two benchmark results."""
    def pct_change(before: float, after: float) -> float:
        return round((after - before) / before * 100, 1) if before else 0.0

    return {
        "baseline": baseline,
        "optimized": optimized,
        "p50_change_pct": pct_change(baseline["p50_ms"], optimized["p50_ms"]),
        "p95_change_pct": pct_change(baseline["p95_ms"], optimized["p95_ms"]),
        "p99_change_pct": pct_change(baseline["p99_ms"], optimized["p99_ms"]),
        "throughput_change_pct": pct_change(
            baseline["throughput_rps"], optimized["throughput_rps"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an ONNX model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default=None,
                        help="Baseline model to compare against")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = benchmark_model(args.model, runs=args.runs, warmup=args.warmup)
    if args.baseline:
        baseline = benchmark_model(args.baseline, runs=args.runs, warmup=args.warmup)
        result = compare_models(baseline, result)
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
