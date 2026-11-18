"""Configuration for model optimization (export, quantization, benchmarks)."""

from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 3
    max_length: int = 128
    opset_version: int = 14
    quantize: str = "none"  # none | int8 | fp16
    benchmark_runs: int = 100
    benchmark_warmup: int = 10
    batch_size: int = 1


@dataclass
class OptimizationSettings:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
