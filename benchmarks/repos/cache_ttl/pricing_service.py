"""Price lookups for the storefront, backed by Cache."""

from .cache import Cache

_cache = Cache()


def get_price(sku: str) -> float:
    cached = _cache.get(sku)
    if cached is not None:
        return cached
    price = _load_price_from_db(sku)
    _cache.set(sku, price)
    return price


def _load_price_from_db(sku: str) -> float:
    raise NotImplementedError("database call stubbed for tests")
