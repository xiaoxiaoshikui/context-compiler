"""Order event producer, unrelated to consumer idempotency."""

def publish(event: dict) -> None:
    raise NotImplementedError("queue publish stubbed for tests")
