"""Order event consumer."""

_SEEN_EVENT_IDS: set[str] = set()


def handle_event(event: dict) -> None:
    event_id = event["event_id"]
    if event_id in _SEEN_EVENT_IDS:
        return
    _SEEN_EVENT_IDS.add(event_id)
    _apply(event)


def _apply(event: dict) -> None:
    raise NotImplementedError("order state transition stubbed for tests")
