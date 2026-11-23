"""Gate-closure state features phi_D for the end-of-day SoC value (v2).

phi_D is built only from what is known at gate closure of D (D-1 12:00): the forecast for D
(horizon 1), the forecast for D+1 (horizon 2) and the calendar. Realised prices never enter
phi_D; they only enter the reward. tests/test_rlvalue.py checks this by perturbing prices.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from bess.data import calendar as cal
from bess.forecast.xgb import DE_HOLIDAYS

PEAK_HOURS = 2.0   # top / bottom window of the forecast statistics (the battery's duration at full power)

STAT_NAMES = ["mean", "std", "top_mean", "bottom_mean", "spread", "peak_valley"]
FEATURE_NAMES = ([f"d0_{s}" for s in STAT_NAMES] + [f"d1_{s}" for s in STAT_NAMES]
                 + ["dow_sin", "dow_cos", "month_sin", "month_cos", "holiday_d0", "holiday_d1",
                    "weekend_d1", "qh_flag"])


def forecast_stats(f: np.ndarray, dt_h: float) -> list[float]:
    """mean, std, mean of the k highest / lowest periods (k = PEAK_HOURS of steps), max - min,
    and top_mean - bottom_mean (the spread a full cycle could capture)."""
    f = np.asarray(f, float)
    k = max(1, int(round(PEAK_HOURS / dt_h)))
    s = np.sort(f)
    top, bottom = float(s[-k:].mean()), float(s[:k].mean())
    return [float(f.mean()), float(f.std()), top, bottom, float(s[-1] - s[0]), top - bottom]


def state_features(day, f_d0: np.ndarray, dt_d0: float, f_d1: np.ndarray, dt_d1: float) -> np.ndarray:
    """phi_D from the gate-closure forecasts of D (f_d0) and D+1 (f_d1)."""
    day = cal.to_date(day)
    nxt = day + dt.timedelta(days=1)
    dow, month = day.weekday(), day.month
    cal_feats = [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                 np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
                 float(day in DE_HOLIDAYS), float(nxt in DE_HOLIDAYS), float(nxt.weekday() >= 5),
                 float(dt_d0 < 1.0)]
    return np.asarray(forecast_stats(f_d0, dt_d0) + forecast_stats(f_d1, dt_d1) + cal_feats, float)
