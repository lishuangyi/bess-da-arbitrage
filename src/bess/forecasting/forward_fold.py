"""Forward-chaining XGB forecaster for every delivery day (v2).

For the reward cache and the chained backtests every day needs a gate-closure forecast that
is out of sample for the model that made it. One model per (year y, horizon k) is fitted on
all hourly days before 1 January y (early stopping on the latest FOLD_ES_SHARE of them,
inside the fold). A forecast made at gate closure of delivery day D uses the models of
year(D), whose training data end on 31 December of the previous year, i.e. before D-1, so
they only contain published prices. This also holds for horizon 2 (D+1, used by the rolling
MILP and the D+1 features): the model is chosen by the decision day D, not by D+1.
Years with fewer than MIN_FOLD_DAYS earlier days get no model.
"""
from __future__ import annotations

import datetime as dt
import pickle

import numpy as np
import xgboost as xgb

from bess.core import calendar as cal
from bess.forecasting import xgb as xgbmod
from bess.forecasting.base import Forecaster, check_information
from bess.forecasting.xgb import FOLD_ES_SHARE, MIN_FOLD_DAYS, dataset, day_features
from bess.input.prices import MarketData


class ForwardFoldXGB(Forecaster):
    name = "xgb"

    def __init__(self, horizons=(1, 2)):
        self.horizons = tuple(horizons)
        self.models: dict[tuple[int, int], xgb.XGBRegressor] = {}
        self.best_iteration: dict[tuple[int, int], int] = {}
        self.fold_days: dict[int, tuple[str, str, int]] = {}   # year -> (first, last, n) history days

    def fit(self, data: MarketData, years: list[int]) -> "ForwardFoldXGB":
        first_day = cal.local_date(data.hour.first_ts)
        for year in sorted(years):
            start = dt.date(year, 1, 1)
            hour = data.hour.truncated_before(cal.day_start_utc(start))
            history = [d for d in cal.daterange(first_day, start - dt.timedelta(days=1)) if hour.is_complete(d)]
            if len(history) < MIN_FOLD_DAYS:
                continue
            n_es = max(14, int(round(FOLD_ES_SHARE * len(history))))
            fit_days, es_days = history[:-n_es], history[-n_es:]
            self.fold_days[year] = (str(history[0]), str(history[-1]), len(history))
            for k in self.horizons:
                Xtr, ytr = dataset(hour, fit_days, k)
                Xes, yes = dataset(hour, es_days, k)
                m = xgb.XGBRegressor(**xgbmod.XGB_PARAMS)
                m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
                self.models[(year, k)] = m
                self.best_iteration[(year, k)] = int(m.best_iteration)
        return self

    def years(self) -> list[int]:
        return sorted({y for y, _ in self.models})

    def predict(self, day, info: MarketData, resolution: str, horizon: int | None = None) -> np.ndarray:
        check_information(day, info)
        day = cal.to_date(day)
        last_known = cal.local_date(info.hour.last_ts)
        k = horizon or (day - last_known).days
        year = (last_known + dt.timedelta(days=1)).year      # year of the decision day D
        if (year, k) not in self.models:
            raise ValueError(f"no forward-fold model for year {year}, horizon {k} (day {day})")
        X = day_features(info.hour, day, k)
        if X is None:
            raise RuntimeError(f"missing lag data for {day}")
        yh = self.models[(year, k)].predict(X).astype(float)
        if resolution == "hour":
            return yh
        return np.repeat(yh, int(round(1.0 / cal.RESOLUTION_HOURS[resolution])))

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"horizons": self.horizons, "models": self.models, "best_iteration": self.best_iteration,
                         "fold_days": self.fold_days}, f)

    @classmethod
    def load(cls, path) -> "ForwardFoldXGB":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(horizons=d["horizons"])
        obj.models, obj.best_iteration, obj.fold_days = d["models"], d["best_iteration"], d["fold_days"]
        return obj
