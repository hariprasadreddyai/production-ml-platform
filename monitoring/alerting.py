"""Alerting: configurable rules with Slack and email notifiers.

Rules are simple threshold checks over a metrics dict. Notifiers send to
Slack via webhook or email via SMTP. In dry-run mode (default when no
webhook or SMTP config is given) alerts are logged instead of sent, which
keeps local runs safe.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    name: str
    metric: str
    threshold: float
    comparison: str = ">"  # > or <
    severity: str = "warning"
    message: str = ""

    def triggered(self, metrics: dict) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self.comparison == ">":
            return value > self.threshold
        return value < self.threshold


@dataclass
class Alert:
    rule: str
    severity: str
    message: str
    value: float


class SlackNotifier:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        text = f"[{alert.severity.upper()}] {alert.rule}: {alert.message} (value={alert.value})"
        if not self.webhook_url:
            logger.warning("Slack dry-run (no webhook configured): %s", text)
            return False
        response = httpx.post(self.webhook_url, json={"text": text}, timeout=10)
        response.raise_for_status()
        return True


class EmailNotifier:
    def __init__(self, smtp_host: str | None = None, smtp_port: int = 587,
                 username: str | None = None, password: str | None = None,
                 sender: str = "mlserve-alerts@example.com",
                 recipients: list[str] | None = None):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients or []

    def send(self, alert: Alert) -> bool:
        if not self.smtp_host or not self.recipients:
            logger.warning("Email dry-run (no SMTP configured): %s", alert.rule)
            return False
        message = EmailMessage()
        message["Subject"] = f"[{alert.severity.upper()}] MLServe alert: {alert.rule}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(
            f"{alert.message}\nRule: {alert.rule}\nValue: {alert.value}"
        )
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(message)
        return True


class Alerter:
    def __init__(self, rules: list[AlertRule] | None = None,
                 notifiers: list | None = None):
        self.rules = rules or []
        self.notifiers = notifiers or [SlackNotifier(), EmailNotifier()]

    def check(self, metrics: dict) -> list[Alert]:
        fired: list[Alert] = []
        for rule in self.rules:
            if rule.triggered(metrics):
                alert = Alert(
                    rule=rule.name,
                    severity=rule.severity,
                    message=rule.message or f"{rule.metric} crossed {rule.threshold}",
                    value=metrics[rule.metric],
                )
                fired.append(alert)
                for notifier in self.notifiers:
                    try:
                        notifier.send(alert)
                    except Exception as exc:  # noqa: BLE001 - alerting must not crash serving
                        logger.error("Failed to send alert via %s: %s",
                                     type(notifier).__name__, exc)
        return fired


DEFAULT_RULES = [
    AlertRule("high_error_rate", "error_rate", 0.05, ">", "critical",
              "Error rate above 5 percent"),
    AlertRule("high_p99_latency", "p99_latency_ms", 200, ">", "warning",
              "p99 latency above 200ms"),
    AlertRule("high_drift", "drift_psi", 0.25, ">", "warning",
              "Input drift PSI above 0.25"),
]
