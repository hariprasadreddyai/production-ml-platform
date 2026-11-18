"""Tests for inference helpers: lexicon predictor and A/B router."""

from serving.inference import ABRouter, lexicon_predict


def test_lexicon_positive():
    label, confidence = lexicon_predict("I love this, it is excellent")
    assert label == "positive"
    assert confidence >= 0.7


def test_lexicon_negative():
    label, confidence = lexicon_predict("Terrible service, completely broken")
    assert label == "negative"
    assert confidence >= 0.7


def test_lexicon_neutral():
    label, confidence = lexicon_predict("The package arrived on Tuesday")
    assert label == "neutral"


def test_lexicon_mixed_sentiment_is_neutral():
    label, _ = lexicon_predict("I love the design but hate the battery")
    assert label == "neutral"


def test_ab_router_disabled_always_stable():
    router = ABRouter(enabled=False)
    assert {router.route() for _ in range(20)} == {"stable"}


def test_ab_router_split_matches_weights():
    router = ABRouter(stable_weight=0.9, canary_weight=0.1, seed=7)
    outcomes = [router.route() for _ in range(1000)]
    canary_share = outcomes.count("canary") / len(outcomes)
    assert 0.05 < canary_share < 0.15


def test_ab_router_keyed_routing_is_stable():
    router = ABRouter(seed=1)
    assert router.route("user-42") == router.route("user-42")
    routed = {router.route(f"user-{i}") for i in range(100)}
    assert routed == {"stable", "canary"}
