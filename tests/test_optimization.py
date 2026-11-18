"""Tests for model optimization: ONNX export, quantization, benchmarking."""

import numpy as np
import onnxruntime as ort

from optimization.benchmarker import benchmark_model, compare_models
from optimization.onnx_exporter import export_to_onnx
from optimization.quantizer import quantize
from training.trainer import build_model


def tiny_model():
    return build_model("distilbert-base-uncased", num_labels=3, tiny=True)


def test_onnx_export_creates_valid_file(tmp_path):
    out = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model(), out, input_length=32)
    assert out.exists()
    assert out.stat().st_size > 0


def test_onnx_model_runs_in_onnxruntime(tmp_path):
    out = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model(), out, input_length=32)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    result = session.run(
        None,
        {
            "input_ids": np.ones((1, 32), dtype=np.int64),
            "attention_mask": np.ones((1, 32), dtype=np.int64),
        },
    )
    assert result[0].shape == (1, 3)


def test_benchmark_returns_percentiles(tmp_path):
    out = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model(), out, input_length=32)
    report = benchmark_model(str(out), runs=10, warmup=2, seq_length=32)
    assert report["runs"] == 10
    assert report["p50_ms"] > 0
    assert report["p99_ms"] >= report["p50_ms"]
    assert report["throughput_rps"] > 0


def test_compare_models_reports_change():
    baseline = {"p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 25.0, "throughput_rps": 100.0}
    optimized = {"p50_ms": 5.0, "p95_ms": 10.0, "p99_ms": 12.0, "throughput_rps": 200.0}
    report = compare_models(baseline, optimized)
    assert report["p50_change_pct"] == -50.0
    assert report["throughput_change_pct"] == 100.0


def test_int8_quantization(tmp_path):
    out = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model(), out, input_length=32)
    result = quantize(out, "int8")
    assert result["method"] == "int8"
    assert result["size_after_mb"] <= result["size_before_mb"]


def test_quantize_unknown_method_raises(tmp_path):
    out = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model(), out, input_length=32)
    try:
        quantize(out, "int4")
    except ValueError as exc:
        assert "int4" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown method")
