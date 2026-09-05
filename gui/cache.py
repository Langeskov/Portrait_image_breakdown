"""Versioned in-session analysis cache.

Uses an exact SHA-256 image key and a small LRU. The mapping-compatible API lets
the existing MainWindow use the cache without knowing its implementation.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import numpy as np

CACHE_VERSION = "v2"


def image_cache_key(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.data)
    return h.hexdigest()


class AnalysisCache:
    """Small LRU cache for AnalysisBundle instances."""

    def __init__(self, capacity: int = 8):
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str, default=None):
        value = self._items.get(key, default)
        if key in self._items:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, bundle: object) -> None:
        self._items[key] = bundle
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __getitem__(self, key: str):
        value = self._items[key]
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: object) -> None:
        self.put(key, value)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
