"""Price-source interface: how the prices of a market are fetched from a data provider.

A source is registered under the `source.kind` used in configs/markets.yaml. Adding a market from a provider that is
already supported needs only a new entry in that file; a new provider needs one `PriceSource` subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

_SOURCES: dict[str, type["PriceSource"]] = {}


def register_source(kind: str):
    def deco(cls):
        cls.kind = kind
        _SOURCES[kind] = cls
        return cls
    return deco


def get_source(kind: str) -> "PriceSource":
    import bess.input.sources  # noqa: F401  (registers the built-in sources)

    if kind not in _SOURCES:
        raise KeyError(f"unknown price source {kind!r}; registered: {sorted(_SOURCES)}")
    return _SOURCES[kind]()


def registered_sources() -> list[str]:
    import bess.input.sources  # noqa: F401

    return sorted(_SOURCES)


class PriceSource(ABC):
    kind: str = "source"
    resolutions: dict[str, float] = {}   # resolution name -> step length in hours

    @abstractmethod
    def fetch(self, spec, cache_dir: Path, resolution: str):
        """Download (with a raw cache under cache_dir/raw/) and return a `bess.input.prices.PriceSeries`."""

    def fetch_days(self, selector: dict, first, last, cache_dir: Path, resolution: str):
        """Prices of delivery days first..last only (used to fill gaps of the primary source)."""
        raise NotImplementedError(f"source {self.kind!r} cannot fetch day ranges")
