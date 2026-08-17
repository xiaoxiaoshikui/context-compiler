"""Structured logging helpers, unrelated to the database pool."""

import json
import sys


def log(event: str, **fields: object) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
