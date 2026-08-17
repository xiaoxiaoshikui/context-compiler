"""Response serialization helpers, unrelated to rate limiting."""

import json


def to_json(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))
