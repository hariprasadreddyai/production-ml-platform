"""INT8 and FP16 quantization for ONNX models.

INT8 uses ONNX Runtime dynamic quantization (weights to int8), which cuts
model size roughly 4x with a small accuracy cost. FP16 casts weights to
half precision for GPUs. Both produce a new .onnx file and report the
size change.
"""

from __future__ import annotations

import os
from pathlib import Path


def quantize_int8(model_path: str | Path, output_path: str | Path | None = None) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(model_path)
    out = Path(output_path) if output_path else src.with_name(
        src.stem + ".int8.onnx"
    )
    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    return out


def quantize_fp16(model_path: str | Path, output_path: str | Path | None = None) -> Path:
    import onnx
    from onnxruntime.transformers import float16

    src = Path(model_path)
    out = Path(output_path) if output_path else src.with_name(
        src.stem + ".fp16.onnx"
    )
    model = onnx.load(str(src))
    fp16_model = float16.convert_float_to_float16(model)
    onnx.save(fp16_model, str(out))
    return out


def quantize(model_path: str | Path, method: str,
             output_path: str | Path | None = None) -> dict:
    """Quantize a model and report before/after sizes.

    method is one of: int8, fp16.
    """
    if method not in ("int8", "fp16"):
        raise ValueError(f"Unknown quantization method: {method}")
    src = Path(model_path)
    if method == "int8":
        out = quantize_int8(src, output_path)
    else:
        out = quantize_fp16(src, output_path)
    return {
        "method": method,
        "input_path": str(src),
        "output_path": str(out),
        "size_before_mb": round(os.path.getsize(src) / 1e6, 2),
        "size_after_mb": round(os.path.getsize(out) / 1e6, 2),
    }
