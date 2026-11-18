"""Tests for monitoring: metrics collector and alerting."""

from monitoring.alerting import (
    DEFAULT_RULES,
    Alerter,
    AlertRule,
    EmailNotifier,
    SlackNotifier,
)
from monitoring.metrics_collector import (
    REQUEST_COUNT,
    get_metrics,
    render_metrics,
)


def test_metrics_render_contains_defined_metrics():
    REQUEST_COUNT.labels(endpoint="/predict", model_version="v1.0.0").inc()
    payload = render_metrics().decode()
    assert "mlserve_requests_total" in payload
    assert "mlserve_prediction_latency_seconds" in payload


def test_get_metrics_returns_registry():
    assert get_metrics() is not None


def test_alert_rule_triggers_above_threshold():
    rule = AlertRule("high_error_rate", "error_rate", 0.05, ">", "critical")
    assert rule.triggered({"error_rate": 0.08}) is True
    assert rule.triggered({"error_rate": 0.01}) is False
    assert rule.triggered({}) is False


def test_alert_rule_less_than_comparison():
    rule = AlertRule("low_throughput", "throughput_rps", 10, "<", "warning")
    assert rule.triggered({"throughput_rps": 5}) is True
    assert rule.triggered({"throughput_rps": 50}) is False


def test_alerter_fires_matching_rules():
    alerter = Alerter(rules=DEFAULT_RULES, notifiers=[])
    fired = alerter.check({"error_rate": 0.10, "p99_latency_ms": 50, "drift_psi": 0.1})
    assert [a.rule for a in fired] == ["high_error_rate"]


def test_alerter_quiet_when_metrics_healthy():
    alerter = Alerter(rules=DEFAULT_RULES, notifiers=[])
    assert alerter.check({"error_rate": 0.01, "p99_latency_ms": 20, "drift_psi": 0.05}) == []


def test_slack_notifier_dry_run_without_webhook():
    notifier = SlackNotifier(webhook_url=None)
    from monitoring.alerting import Alert

    assert notifier.send(Alert("test", "warning", "msg", 1.0)) is False


def test_email_notifier_dry_run_without_smtp():
    notifier = EmailNotifier()
    from monitoring.alerting import Alert

    assert notifier.send(Alert("test", "warning", "msg", 1.0)) is False
