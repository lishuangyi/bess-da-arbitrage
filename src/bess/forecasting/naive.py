"""Seasonal-naive forecaster: price of the same local time-of-day one week earlier."""
from __future__ import annotations

import datetime as dt

import numpy as np

from bess.core import calendar as cal
from bess.forecasting.base import Forecaster, check_information, profile_on_grid
from bess.input.prices import MarketData


class SeasonalNaive(Forecaster):
    """D-7 same slot at native resolution.

    Falls back to D-14 (then D-21) if D-7 is incomplete or not yet published
    (e.g. two days ahead in rolling optimisation this never happens for lag 7).
    If the target is 15-min but the source week precedes the 15-min market,
    the hourly source is forward-filled to 15 min.
    """

    name = "naive"

    def __init__(self, lag_days: int = 7, max_fallback_weeks: int = 3):
        self.lag_days = lag_days
        self.max_fallback_weeks = max_fallback_weeks

    def predict(self, day, info: MarketData, resolution: str) -> np.ndarray:
        check_information(day, info)
        day = cal.to_date(day)
        dt_h = cal.RESOLUTION_HOURS[resolution]
        for k in range(1, self.max_fallback_weeks + 1):
            src_day = day - dt.timedelta(days=self.lag_days * k)
            for src in (info.series(resolution), info.hour):
                if src.empty or src.first_ts > cal.day_start_utc(src_day):
                    continue
                prof = profile_on_grid(src, src_day, day, dt_h)
                if prof is not None:
                    return prof
        raise RuntimeError(f"no source day available for naive forecast of {day}")
