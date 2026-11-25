"""Forecaster interface and helpers.

A forecaster receives the information set available at DA gate closure of the
delivery day being planned (`MarketData.known_at(...)`) and returns prices on
the calendar grid of the target day. It never sees the target day's prices.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bess.core import calendar as cal
from bess.input.prices import MarketData, PriceSeries


class Forecaster(ABC):
    name = "forecaster"

    @abstractmethod
    def predict(self, day, info: MarketData, resolution: str) -> np.ndarray:
        """Price forecast for delivery day `day` at `resolution` given information `info`."""


def check_information(day, info: MarketData) -> None:
    """Guard: the information set must not contain any price of `day` or later."""
    last = info.latest_timestamp()
    if last is not None and last >= cal.day_start_utc(day):
        raise RuntimeError(f"look-ahead: information up to {last} used to forecast {day}")


def profile_on_grid(src: PriceSeries, src_day, target_day, target_dt_h: float) -> np.ndarray | None:
    """Map the prices of `src_day` onto the grid of `target_day` by local wall-clock time.

    For every target period, take the source price of the latest source period
    whose local start minute is <= the target's start minute. This forward-fills
    hourly sources onto a 15-min grid, fills the hour missing on a spring DST
    source day from the previous hour, and reuses 02:00-03:00 on an autumn DST
    target day. Returns None if the source day is incomplete.
    """
    p = src.day_prices(src_day)
    if not np.isfinite(p).all():
        return None
    s_min = cal.day_local_minutes(src_day, src.dt_h)
    t_min = cal.day_local_minutes(target_day, target_dt_h)
    # first occurrence of each wall-clock minute (autumn DST source day), sorted by minute
    uniq, first = np.unique(s_min, return_index=True)
    vals = p[first]
    pos = np.clip(np.searchsorted(uniq, t_min, side="right") - 1, 0, len(uniq) - 1)
    return vals[pos]
