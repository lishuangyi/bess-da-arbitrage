"""Market registry: which markets exist, where their prices come from and what their timing rules are.

Markets are declared in configs/markets.yaml; experiment configs name one with `market: <id>`. A market is
identified by its id (e.g. `de_lu`), which is distinct from the provider's selector (SMARD filter and API region:
Austria is filter 4170 under the API region `DE-LU`).

Supported today: day-ahead markets whose delivery days and gate closure are defined in Europe/Berlin local time
(every bidding zone of the coupled European day-ahead auction in CET) and whose calendar features use German
holidays (what the forecaster was built with). Other settings are rejected explicitly - see `_check_supported` -
rather than silently mis-timed.

Multi-market inputs: `load_inputs(target, as_of_utc, exogenous)` returns the information set of several markets at
one decision instant, each filtered with its own publication rule. Forecasters that use other markets consume it
(`bess.forecasting.multimarket`).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from bess.core import calendar as cal
from bess.core.config import REPO_ROOT, load_yaml

REGISTRY_FILE = "markets.yaml"
SUPPORTED_TIMEZONE = cal.TZ
SUPPORTED_HOLIDAYS = "DE"


@dataclass(frozen=True)
class MarketSpec:
    id: str
    name: str
    product: str                      # "day_ahead" (the only market product with a scheduling model today)
    source: dict                      # provider selector, e.g. {kind: smard, base_url, filter_id, region}
    timezone: str                     # delivery days and gate closure are local days/times in this zone
    gate_closure_local: str           # on D-1
    publication_local: str            # conservative publication time of D's prices, on D-1
    qh_market_start: str              # first delivery day with native 15-min products
    download_start: str
    cache_dir: str                    # relative to the repository root
    holidays: str = "DE"              # holiday calendar of the forecaster's calendar features
    fallback: dict | None = None      # second source for periods the primary one leaves empty (checked on overlap)
    currency: str = "EUR"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cache_path(self) -> Path:
        return REPO_ROOT / self.cache_dir

    @property
    def qh_start_date(self) -> dt.date:
        return cal.to_date(self.qh_market_start)

    def day_ahead_market(self):
        """The market rules object used by the MILP, the RL environment and settlement."""
        if self.product != "day_ahead":
            raise NotImplementedError(f"market {self.id!r}: product {self.product!r} has no scheduling model")
        from bess.core.markets.day_ahead import DayAheadMarket
        return DayAheadMarket(self.gate_closure_local)

    def gate_closure_utc(self, delivery_day) -> dt.datetime:
        return cal.gate_closure_utc(delivery_day, self.gate_closure_local)


_FIELDS = {f for f in MarketSpec.__dataclass_fields__}


def _check_supported(spec: MarketSpec) -> None:
    if spec.timezone != SUPPORTED_TIMEZONE:
        raise NotImplementedError(
            f"market {spec.id!r}: timezone {spec.timezone!r} is not supported yet; delivery days, gate closure and "
            f"publication are computed in {SUPPORTED_TIMEZONE} (bess/core/calendar.py). Thread a per-market calendar "
            "through PriceSeries, MarketData.known_at and the forecast grids before adding it.")
    if spec.holidays != SUPPORTED_HOLIDAYS:
        raise NotImplementedError(
            f"market {spec.id!r}: holiday calendar {spec.holidays!r} is not supported yet; the forecaster's "
            f"calendar features use {SUPPORTED_HOLIDAYS} holidays (bess/forecasting/xgb.py).")


@lru_cache(maxsize=None)
def _registry(path: str) -> dict[str, MarketSpec]:
    raw = load_yaml(path)["markets"]
    out = {}
    for mid, d in raw.items():
        known = {k: v for k, v in d.items() if k in _FIELDS}
        extra = {k: v for k, v in d.items() if k not in _FIELDS}
        out[mid] = MarketSpec(id=mid, extra=extra, **known)
    return out


def registry(path: str = REGISTRY_FILE) -> dict[str, MarketSpec]:
    """All declared markets (including ones whose source or calendar is not supported yet)."""
    return dict(_registry(str(path)))


def get_market(market_id: str) -> MarketSpec:
    reg = registry()
    if market_id not in reg:
        raise KeyError(f"unknown market {market_id!r}; declared in configs/{REGISTRY_FILE}: {sorted(reg)}")
    spec = reg[market_id]
    _check_supported(spec)
    return spec


def resolve_market(market) -> MarketSpec:
    """MarketSpec from a spec, a market id, or an experiment config with a `market:` key."""
    if isinstance(market, MarketSpec):
        _check_supported(market)
        return market
    if isinstance(market, str):
        return get_market(market)
    if isinstance(market, dict):
        if "market" in market:
            return get_market(market["market"])
        if "base_config" in market:   # a v2 profile: the market of its base experiment config
            from bess.core.config import experiment_config
            return resolve_market(experiment_config(market["base_config"]))
    raise TypeError(f"cannot resolve a market from {type(market).__name__}")


# --------------------------------------------------------------------------- multi-market information sets

@dataclass(frozen=True)
class MarketInputs:
    """Information available at one decision instant, for a target market and optional exogenous markets.

    Every market is cut with its own publication rule at the same `as_of_utc`; timestamps are UTC, so series of
    different markets align on the time axis directly.
    """

    target: str
    as_of_utc: dt.datetime
    markets: dict          # market id -> MarketData (already restricted to what was published at as_of_utc)

    @property
    def exogenous(self) -> list[str]:
        return [m for m in self.markets if m != self.target]

    def __getitem__(self, market_id: str):
        return self.markets[market_id]


def load_inputs(target: str, as_of_utc: dt.datetime, exogenous: tuple[str, ...] | list[str] = (),
                data: dict | None = None) -> MarketInputs:
    """Information set of `target` (and `exogenous` markets) at the decision instant `as_of_utc`.

    The decision instant must be explicit: a forecast for D+1 made at the gate closure of D keeps D's cut-off.
    `data` may pass already loaded full MarketData per market id (otherwise loaded from the caches).
    """
    from bess.input.prices import load_market_data

    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    out = {}
    for mid in (target, *[m for m in exogenous if m != target]):
        spec = get_market(mid)
        full = (data or {}).get(mid) or load_market_data(spec)
        out[mid] = full.known_at(as_of_utc, spec.publication_local)
    return MarketInputs(target, as_of_utc, out)
