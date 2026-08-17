"""Client for the downstream recommendation service."""

from .backoff import call_with_backoff


class RecommendationClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get_recommendations(self, user_id: str) -> list[str]:
        return call_with_backoff(lambda: self._request(user_id))

    def _request(self, user_id: str) -> list[str]:
        raise NotImplementedError("network call stubbed for tests")
