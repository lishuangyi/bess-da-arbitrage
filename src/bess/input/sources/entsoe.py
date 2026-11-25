"""ENTSO-E Transparency Platform (declared, not implemented).

Reserved so that markets outside SMARD's coverage (e.g. FR, NL, BE, PL day-ahead) can be added in
configs/markets.yaml with `source: {kind: entsoe, area: <EIC code>}`. The Transparency API needs a personal security
token and returns XML documents; the parser must map them onto the same UTC grid as `PriceSeries` (NaN for missing
periods, no interpolation). Until that exists, using such a market fails loudly instead of returning invented data.
"""
from __future__ import annotations

from pathlib import Path

from bess.input.sources.base import PriceSource, register_source


@register_source("entsoe")
class EntsoeSource(PriceSource):
    resolutions = {"hour": 1.0, "quarterhour": 0.25}

    def fetch(self, spec, cache_dir: Path, resolution: str):
        raise NotImplementedError(
            f"market {spec.id!r}: the ENTSO-E source is declared but not implemented yet "
            "(bess/input/sources/entsoe.py); implement EntsoeSource.fetch to enable it")
