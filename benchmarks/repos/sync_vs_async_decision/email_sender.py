"""Enqueues transactional emails for asynchronous delivery."""

from .queue_client import enqueue


def send_receipt_email(order_id: str, to_address: str) -> None:
    enqueue("send_email", order_id=order_id, to_address=to_address)
