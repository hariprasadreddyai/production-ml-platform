"""Training pipeline: distributed PyTorch training with MLflow tracking."""

from training.config import Config
from training.trainer import train

__all__ = ["Config", "train"]
