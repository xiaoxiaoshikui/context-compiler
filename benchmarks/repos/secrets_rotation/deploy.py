"""Deployment entrypoint for the billing webhook service."""

from .secrets_manager import fetch_secret


def build_env(stage: str) -> dict:
    return {
        "BILLING_API_KEY": fetch_secret(f"{stage}/billing-webhook/api-key"),
        "STAGE": stage,
    }
