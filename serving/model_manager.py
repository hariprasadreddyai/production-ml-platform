"""Model manager: versioned models, hot reloading, A/B routing."""

from __future__ import annotations

import logging

from serving.inference import ABRouter, HFModel, StubPredictor

logger = logging.getLogger(__name__)


class ModelManager:
    """Holds the stable and canary predictors and routes between them."""

    def __init__(self, stable=None, canary=None,
                 stable_version: str = "v1.0.0",
                 canary_version: str | None = "v1.1.0-canary",
                 router: ABRouter | None = None):
        self.models: dict[str, StubPredictor | HFModel | None] = {
            "stable": stable or StubPredictor(),
            "canary": canary,
        }
        self.versions = {
            "stable": stable_version,
            "canary": canary_version,
        }
        self.router = router or ABRouter()

    @property
    def canary_enabled(self) -> bool:
        return self.models["canary"] is not None and self.router.enabled

    def predict(self, text: str, request_id: str | None = None) -> dict:
        slot = self.router.route(request_id or text)
        model = self.models.get(slot) or self.models["stable"]
        if slot == "canary" and model is None:
            slot = "stable"
            model = self.models["stable"]
        # The stable slot always holds a predictor by construction.
        assert model is not None
        label, confidence = model.predict(text)
        return {
            "label": label,
            "confidence": confidence,
            "model_version": self.versions[slot],
            "slot": slot,
        }

    def reload(self, slot: str, model_path: str, version: str) -> bool:
        """Hot-reload one slot. model_path 'stub' loads the stub predictor."""
        if slot not in ("stable", "canary"):
            raise ValueError(f"Unknown slot: {slot}")
        if model_path == "stub":
            predictor: StubPredictor | HFModel = StubPredictor()
        else:
            predictor = HFModel(model_path)
        self.models[slot] = predictor
        self.versions[slot] = version
        logger.info("Reloaded %s slot to version %s", slot, version)
        return True
