"""Gradient-boosted day-ahead price forecaster (hourly model).

Features for target delivery day T at forecast horizon k (k=1: T is the day
being bid at gate closure; k=2: the day after, used by the 48 h rolling MILP).
The latest fully published day at gate closure is T-k, so only days <= T-k
are used:
- lagged prices at the same local time on T-k, T-k-1 and T-7 (T-7 <= T-k for k <= 7);
- level/shape of day T-k: mean, min, max, std;
- calendar: local hour, day of week, month, day-of-year (sin/cos),
  weekend flag, German public holiday flag (national holidays).
One model per horizon, trained on the hourly training split; early stopping on
the validation split. For a 15-min target the hourly forecast is forward-filled
to the quarter-hours (v1).
"""
from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import xgboost as xgb

from bess.data import calendar as cal
from bess.data.smard import MarketData, PriceSeries
from bess.forecast.base import Forecaster, check_information, profile_on_grid

DE_HOLIDAYS = holidays.country_holidays("DE", years=range(2018, 2031))

FEATURES = ["lag_a", "lag_b", "lag_7", "prev_mean", "prev_min", "prev_max", "prev_std",
            "hour", "dow", "month", "doy_sin", "doy_cos", "weekend", "holiday"]

XGB_PARAMS = dict(n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  early_stopping_rounds=100, n_jobs=6, random_state=0)


def day_features(hour: PriceSeries, target, k: int) -> np.ndarray | None:
    """Feature matrix (n_hours(target) x len(FEATURES)) for hourly target day; None if lags missing."""
    target = cal.to_date(target)
    a = profile_on_grid(hour, target - dt.timedelta(days=k), target, 1.0)
    b = profile_on_grid(hour, target - dt.timedelta(days=k + 1), target, 1.0)
    w = profile_on_grid(hour, target - dt.timedelta(days=7), target, 1.0)
    if a is None or b is None or w is None:
        return None
    prev = hour.day_prices(target - dt.timedelta(days=k))
    n = cal.expected_steps(target, 1.0)
    hod = cal.day_local_minutes(target, 1.0) // 60
    doy = target.timetuple().tm_yday
    dow = target.weekday()
    const = [prev.mean(), prev.min(), prev.max(), prev.std(), ]
    X = np.column_stack([
        a, b, w,
        *[np.full(n, c) for c in const],
        hod, np.full(n, dow), np.full(n, target.month),
        np.full(n, np.sin(2 * np.pi * doy / 365.25)), np.full(n, np.cos(2 * np.pi * doy / 365.25)),
        np.full(n, float(dow >= 5)), np.full(n, float(target in DE_HOLIDAYS)),
    ])
    return X


def dataset(hour: PriceSeries, days: list, k: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for d in days:
        y = hour.day_prices(d)
        if not np.isfinite(y).all():
            continue
        X = day_features(hour, d, k)
        if X is None:
            continue
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


class XGBForecaster(Forecaster):
    name = "xgb"

    def __init__(self, horizons=(1, 2)):
        self.horizons = tuple(horizons)
        self.models: dict[int, xgb.XGBRegressor] = {}
        self.best_iteration: dict[int, int] = {}

    def fit(self, data: MarketData, train_days: list, val_days: list) -> "XGBForecaster":
        """Train on hourly prices. Training targets/lags never extend past the last train day."""
        last_train = max(train_days)
        train_hour = data.hour.truncated_before(cal.day_end_utc(last_train))
        for k in self.horizons:
            Xtr, ytr = dataset(train_hour, train_days, k)
            Xva, yva = dataset(data.hour, val_days, k)
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            self.models[k] = m
            self.best_iteration[k] = int(m.best_iteration)
        return self

    def predict(self, day, info: MarketData, resolution: str, horizon: int | None = None) -> np.ndarray:
        """Forecast for `day`. The horizon is inferred from the information set:
        k = (day - last fully known day)."""
        check_information(day, info)
        day = cal.to_date(day)
        last_known = cal.local_date(info.hour.last_ts)
        k = horizon or (day - last_known).days
        if k not in self.models:
            raise ValueError(f"no model for horizon {k} (day {day}, last known {last_known})")
        X = day_features(info.hour, day, k)
        if X is None:
            raise RuntimeError(f"missing lag data for {day}")
        yh = self.models[k].predict(X).astype(float)
        if resolution == "hour":
            return yh
        # forward-fill the hourly forecast to the 15-min grid (UTC hours align with
        # quarter-hours, also on DST days: 23*4 = 92, 25*4 = 100)
        reps = int(round(1.0 / cal.RESOLUTION_HOURS[resolution]))
        return np.repeat(yh, reps)

    def save(self, path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "best_iteration": self.best_iteration}, f)

    @classmethod
    def load(cls, path) -> "XGBForecaster":
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(horizons=tuple(d["models"]))
        obj.models, obj.best_iteration = d["models"], d["best_iteration"]
        return obj


# Share of a fold's history (the latest days) used for early stopping inside the fold.
FOLD_ES_SHARE = 0.15
MIN_FOLD_DAYS = 60


def forward_fold_forecasts(data: MarketData, days: list, k: int = 1, history_days: list | None = None) -> dict:
    """Forward-chaining (expanding-window) out-of-sample hourly forecasts.

    Each day of calendar year y is predicted by a model fitted only on `history_days`
    before 1 January y (default: `days` itself). Early stopping uses the latest
    FOLD_ES_SHARE of those history days, so it stays inside the fold. A year with
    fewer than MIN_FOLD_DAYS history days (e.g. the first year) gets no forecasts.
    Lag features only read days <= T-k, as for the gate-closure forecaster.
    Used for RL training observations (no in-sample forecasts, no later data).
    """
    days = sorted(cal.to_date(d) for d in days)
    history = sorted(cal.to_date(d) for d in (history_days if history_days is not None else days))
    feat_hour = data.hour.truncated_before(cal.day_end_utc(days[-1]))
    out = {}
    for year in sorted({d.year for d in days}):
        start = dt.date(year, 1, 1)
        fold = [d for d in history if d < start]
        if len(fold) < MIN_FOLD_DAYS:
            continue
        n_es = max(14, int(round(FOLD_ES_SHARE * len(fold))))
        fit_days, es_days = fold[:-n_es], fold[-n_es:]
        hour = data.hour.truncated_before(cal.day_start_utc(start))     # nothing from year y or later
        Xtr, ytr = dataset(hour, fit_days, k)
        Xes, yes = dataset(hour, es_days, k)
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
        for d in (d for d in days if d.year == year):
            X = day_features(feat_hour, d, k)
            if X is not None:
                out[d] = m.predict(X).astype(float)
    return out
