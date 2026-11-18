"""Export a PyTorch model to ONNX for portable, fast inference.

Usage:
    python -m optimization.onnx_exporter --model models/latest.pt
    python -m optimization.onnx_exporter --model models/latest.pt \
        --output models/latest.onnx --opset 14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_to_onnx(model, output_path: str | Path, input_length: int = 128,
                   opset_version: int = 14, dynamic_batch: bool = True) -> Path:
    """Export a torch module to ONNX with input_ids and attention_mask."""
    model.eval()
    dummy_ids = torch.ones(1, input_length, dtype=torch.long)
    dummy_mask = torch.ones(1, input_length, dtype=torch.long)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    # torch's exporter splits large weights into an external .data file.
    # Consolidate into a single self-contained file so the model is one
    # artifact and its on-disk size stays meaningful for serving,
    # quantization, and benchmarking.
    import onnx

    onnx_model = onnx.load(str(out))
    onnx.load_external_data_for_model(onnx_model, str(out.parent))
    onnx.save(onnx_model, str(out))
    data_file = out.with_name(out.name + ".data")
    if data_file.exists():
        data_file.unlink()
    return out


def export_classifier(model_name: str, num_labels: int, output_path: str,
                      opset_version: int = 14, tiny: bool = False) -> Path:
    """Export a HuggingFace sequence classifier to ONNX."""
    from training.trainer import build_model

    model = build_model(model_name, num_labels, tiny=tiny)
    return export_to_onnx(model, output_path, opset_version=opset_version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a model to ONNX")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. distilbert-base-uncased) or path")
    parser.add_argument("--output", default="models/latest.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--tiny", action="store_true",
                        help="Small random-init model, no downloads")
    args = parser.parse_args()

    path = export_classifier(
        args.model, args.num_labels, args.output,
        opset_version=args.opset, tiny=args.tiny,
    )
    print(f"Exported ONNX model to {path}")


if __name__ == "__main__":
    main()
