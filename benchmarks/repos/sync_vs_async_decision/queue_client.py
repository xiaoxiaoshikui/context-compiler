"""Thin wrapper over the background job queue."""

def enqueue(job_name: str, **kwargs: object) -> None:
    raise NotImplementedError("queue backend stubbed for tests")
