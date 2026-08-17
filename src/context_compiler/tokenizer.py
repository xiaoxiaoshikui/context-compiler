from __future__ import annotations

import re
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, max_tokens: int) -> str: ...


_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)?"
    r"|[^\s]",
    re.UNICODE,
)


class HeuristicTokenCounter:
    """Dependency-free token estimator.

    It deliberately treats each CJK character as a token-like unit and splits
    source-code identifiers/punctuation. It is useful for relative budgeting,
    but not billing-grade token accounting.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_TOKEN_RE.findall(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        matches = list(_TOKEN_RE.finditer(text))
        if len(matches) <= max_tokens:
            return text
        end = matches[max_tokens - 1].end()
        return text[:end].rstrip() + "\n…"


class TiktokenCounter:
    """Optional exact-ish counter for OpenAI-family tokenizers.

    Requires the optional `tiktoken` dependency. The encoding name defaults to
    cl100k_base and can be changed by callers.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "tiktoken is not installed. Install with: pip install -e '.[exact]'"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens]).rstrip() + "\n…"


def make_token_counter(name: str = "heuristic") -> TokenCounter:
    if name == "heuristic":
        return HeuristicTokenCounter()
    if name == "tiktoken":
        return TiktokenCounter()
    raise ValueError(f"Unknown tokenizer: {name}")
