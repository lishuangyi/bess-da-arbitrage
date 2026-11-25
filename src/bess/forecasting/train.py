"""Fit the XGBoost price forecasters.

Usage: python -m bess train xgb [--kind forward-fold|single] [--config ...]
  forward-fold  one model per (year, horizon), each fitted only on days before 1 January of that year
                (all learned-value experiments; profile configs/v2*.yaml; written to the profile's `forecaster` path)
  single        one model fitted on the train split of configs/experiment.yaml, early-stopped on its validation split
                (written to <models_dir>/xgb.pkl; used by `bess benchmark public`)
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from bess.core import calendar as cal
from bess.core.config import experiment_config, forecaster_path, models_dir, v2_config
from bess.forecasting.forward_fold import ForwardFoldXGB
from bess.forecasting.xgb import XGBForecaster
from bess.input.prices import MarketData, load_market_data


def get_xgb(cfg: dict, data: MarketData, refit: bool = False) -> XGBForecaster:
    """Single XGB of an experiment config: loaded from <models_dir>/xgb.pkl, fitted if missing or `refit`."""
    from bess.input.periods import all_split_days, period_days

    path = models_dir(cfg) / "xgb.pkl"
    if path.exists() and not refit:
        return XGBForecaster.load(path)
    train_days = all_split_days(cfg, data, "train")   # hourly targets exist for every segment
    val_days, _, _ = period_days(cfg, data, "validation")
    print(f"fitting XGB on {len(train_days)} train days (early stopping on {len(val_days)} validation days)")
    fc = XGBForecaster(horizons=(1, 2)).fit(data, train_days, val_days)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.save(path)
    print(f"  best iterations: {fc.best_iteration}")
    return fc


def get_forward_fold(v2: dict, data: MarketData, years: list[int], refit: bool = False, path=None) -> ForwardFoldXGB:
    """Forward-fold XGB of a v2 profile (shared by all profiles of one market: forecasts do not depend on the
    battery). Loaded if present, fitted for `years` otherwise or if `refit`."""
    path = path or forecaster_path(v2)
    if path.exists() and not refit:
        return ForwardFoldXGB.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"fitting forward-fold XGB for years {years} ...", flush=True)
    fc = ForwardFoldXGB().fit(data, years)
    fc.save(path)
    print(f"  {len(fc.models)} models in {(time.perf_counter() - t0) / 60:.1f} min; folds: {fc.fold_days}", flush=True)
    return fc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["forward-fold", "single"], default="forward-fold")
    ap.add_argument("--config", default=None,
                    help="forward-fold: v2 profile (default v2.yaml); single: experiment config (default experiment.yaml)")
    ap.add_argument("--out", default=None, help="forward-fold only: write the models here instead of the profile path")
    args = ap.parse_args(argv)
    if args.kind == "single":
        cfg = experiment_config(args.config or "experiment.yaml")
        get_xgb(cfg, load_market_data(cfg), refit=True)
        return
    from pathlib import Path

    from bess.input.periods import period_days_res

    v2, base = v2_config(args.config or "v2.yaml")
    data = load_market_data(base)
    days_res, _ = period_days_res(v2, base, data, "cache")
    # every year of the cache period, plus the year of the next delivery day, so that an ex-ante forecast right after
    # the last cached day (e.g. 1 January) has a model; each fold only uses days before its own 1 January
    next_day = cal.local_date(data.hour.last_ts) + dt.timedelta(days=1)
    years = sorted({d.year for d, _ in days_res} | {next_day.year})
    get_forward_fold(v2, data, years, refit=True, path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
