"""Customer account model."""

from dataclasses import dataclass


@dataclass
class Account:
    id: str
    email: str
    deleted_at: str | None = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
