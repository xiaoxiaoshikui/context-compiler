"""Checkout page rendering."""

from .flags import is_enabled


def render_checkout(user_id: str) -> str:
    if is_enabled("new_checkout_flow", user_id=user_id):
        return "checkout_v2.html"
    return "checkout_v1.html"
