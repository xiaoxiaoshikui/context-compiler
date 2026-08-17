"""Checkout funnel analytics events, unrelated to flag evaluation."""

def track(event: str, user_id: str, **props: object) -> None:
    raise NotImplementedError("analytics sink stubbed for tests")
