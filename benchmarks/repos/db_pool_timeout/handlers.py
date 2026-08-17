"""HTTP handlers for the orders API."""

from .db import get_connection


def create_order(payload: dict) -> dict:
    with get_connection() as conn:
        order_id = conn.insert("orders", payload)
    return {"order_id": order_id}


def get_order(order_id: str) -> dict:
    with get_connection() as conn:
        return conn.select_one("orders", order_id)
