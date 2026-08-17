"""Thin client over the managed secrets store."""


def fetch_secret(path: str) -> str:
    raise NotImplementedError("secrets manager client stubbed for tests")


def rotate_secret(path: str) -> str:
    raise NotImplementedError("secrets manager client stubbed for tests")
