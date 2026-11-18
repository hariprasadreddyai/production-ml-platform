"""Configuration loading for the training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 3
    max_length: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    distributed: bool = False


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "text-classifier"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            mlflow=MLflowConfig(**raw.get("mlflow", {})),
        )
