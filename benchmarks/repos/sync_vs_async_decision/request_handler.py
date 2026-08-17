"""HTTP request handler for order checkout, unrelated to queue internals."""

from .email_sender import send_receipt_email


def handle_checkout_request(order: dict) -> dict:
    send_receipt_email(order["id"], order["email"])
    return {"status": "ok"}
