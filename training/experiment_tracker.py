"""Experiment tracking wrapper around MLflow.

Keeps the trainer free of MLflow boilerplate and degrades gracefully:
if MLflow is not installed or the tracking server is unreachable, logging
becomes a no-op instead of crashing a training run.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._run = None
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            logger.warning("MLflow unavailable, tracking disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def __enter__(self) -> "ExperimentTracker":
        if self._mlflow is not None:
            self._run = self._mlflow.start_run()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._mlflow is not None and self._run is not None:
            self._mlflow.end_run()
            self._run = None

    def log_params(self, params: dict) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model") -> None:
        if self._mlflow is not None:
            self._mlflow.pytorch.log_model(model, artifact_path=artifact_path)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
