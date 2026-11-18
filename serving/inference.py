"""Inference: predictors and deterministic A/B traffic routing.

Predictors share one interface: predict(text) -> (label, confidence).
StubPredictor is a lexicon-based sentiment classifier used for local demos,
tests, and CI when no trained model is available. HFModel wraps a real
HuggingFace sequence classifier when one is.
"""

from __future__ import annotations

import hashlib
import random
import re

LABELS = ["negative", "neutral", "positive"]

POSITIVE_WORDS = {
    "love", "loved", "great", "excellent", "fantastic", "superb", "wonderful",
    "best", "amazing", "awesome", "good", "fast", "faster", "friendly",
    "recommended", "happy", "improved", "solid", "easy", "solved", "fixed",
    "perfect", "brilliant", "enjoy", "satisfied", "helpful", "smooth",
}

NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "worst", "bad", "broken", "slow",
    "crashes", "crash", "disappointed", "disappointing", "nightmare",
    "waste", "useless", "unusable", "horrible", "poor", "bug", "bugs",
    "failed", "failure", "annoying", "frustrating", "overpriced", "rude",
}


def lexicon_predict(text: str) -> tuple[str, float]:
    """Rule-based sentiment prediction with a calibrated confidence."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    score = len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)
    if score > 0:
        label = "positive"
        confidence = min(0.6 + 0.1 * score, 0.95)
    elif score < 0:
        label = "negative"
        confidence = min(0.6 + 0.1 * abs(score), 0.95)
    else:
        label = "neutral"
        confidence = 0.65
    return label, round(confidence, 2)


class StubPredictor:
    """Offline sentiment predictor for demos, tests, and CI."""

    version = "stub-1.0.0"

    def predict(self, text: str) -> tuple[str, float]:
        return lexicon_predict(text)


class HFModel:
    """Wrapper around a HuggingFace sequence classification model."""

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._torch = torch
        self.version = model_path

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return LABELS[idx], round(float(probs[idx]), 4)


class ABRouter:
    """Deterministic A/B traffic splitter.

    route(key) hashes the key (request id or text) so the same input always
    lands on the same variant. route() with no key uses a seeded RNG.
    """

    def __init__(self, stable_weight: float = 0.9, canary_weight: float = 0.1,
                 enabled: bool = True, seed: int = 42):
        self.stable_weight = stable_weight
        self.canary_weight = canary_weight
        self.enabled = enabled
        self._rng = random.Random(seed)

    def route(self, key: str | None = None) -> str:
        if not self.enabled or self.canary_weight <= 0:
            return "stable"
        if key is not None:
            digest = hashlib.md5(key.encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
        else:
            value = self._rng.random()
        return "stable" if value < self.stable_weight else "canary"
