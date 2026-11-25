"""Gate-closure price forecast of one delivery day (what the scheduler would have seen).

Usage: python -m bess forecast [--config v2_isea.yaml] [--day YYYY-MM-DD] [--model PATH] [--out CSV]
The information set is cut at the gate closure of the day (D-1 12:00 local, prices published by then only), the
forward-fold XGB model of the decision year forecasts the day at its native resolution. Without --day the next
delivery day after the latest cached prices is forecast (an ex-ante forecast); for a past day the realised prices
and the forecast error are shown too. Writes results/forecasts/<market>_<day>.csv by default.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from bess.core import calendar as cal
from bess.core.config import REPO_ROOT, forecaster_path, v2_config
from bess.forecasting.forward_fold import ForwardFoldXGB
from bess.input.markets import resolve_market
from bess.input.prices import load_market_data


def forecast_day(v2: dict, base: dict, day: dt.date, model_path: Path | None = None) -> pl.DataFrame:
    spec = resolve_market(base)
    data = load_market_data(spec)
    path = model_path or forecaster_path(v2)
    if not Path(path).exists():
        raise SystemExit(f"{path} missing - fit it with `python -m bess train xgb` "
                         "(or `python -m bess cache reward`, which fits it if missing)")
    fc = ForwardFoldXGB.load(path)
    res = "quarterhour" if day >= spec.qh_start_date else "hour"
    info = data.known_at(spec.gate_closure_utc(day), spec.publication_local)
    try:
        yhat = fc.predict(day, info, res)
    except RuntimeError as e:   # missing lag days: the forecaster never interpolates
        lags = [day - dt.timedelta(days=k) for k in (1, 2, 7)]
        gaps = [str(d) for d in lags if not info.hour.is_complete(d)]
        raise SystemExit(f"cannot forecast {day}: lag days {gaps} are incomplete in {spec.cache_dir}/ "
                         f"(run `python -m bess download --market {spec.id}`, or pick another --day)") from e
    ts = cal.day_index(day, cal.RESOLUTION_HOURS[res]).astype("int64")
    df = pl.DataFrame({
        "delivery_start_utc": pl.Series(ts, dtype=pl.Int64).cast(pl.Datetime("ms", "UTC")),
        "forecast_eur_mwh": yhat,
    }).with_columns(pl.col("delivery_start_utc").dt.convert_time_zone(spec.timezone).alias("delivery_start_local"))
    series = data.series(res)
    if series.is_complete(day):
        df = df.with_columns(pl.Series("realised_eur_mwh", series.day_prices(day)))
    return df.select("delivery_start_local", *[c for c in df.columns if c != "delivery_start_local"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="v2_isea.yaml", help="v2 profile (market and forecaster), default v2_isea.yaml")
    ap.add_argument("--day", default=None, help="delivery day (default: the day after the latest cached prices)")
    ap.add_argument("--model", default=None, help="forward-fold model pickle (default: the profile's forecaster)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    v2, base = v2_config(args.config)
    spec = resolve_market(base)
    if args.day:
        day = cal.to_date(args.day)
    else:
        data = load_market_data(spec)
        day = cal.local_date(data.hour.last_ts) + dt.timedelta(days=1)
    df = forecast_day(v2, base, day, Path(args.model) if args.model else None)
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "forecasts" / f"{spec.id}_{day}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, float_precision=2)
    f = df["forecast_eur_mwh"].to_numpy()
    print(f"{spec.id} {day}: {len(f)} steps, forecast mean {f.mean():.1f}, min {f.min():.1f}, max {f.max():.1f} EUR/MWh "
          f"(information cut at gate closure {spec.gate_closure_utc(day).astimezone(cal.ZONE):%Y-%m-%d %H:%M %Z})")
    if "realised_eur_mwh" in df.columns:
        err = f - df["realised_eur_mwh"].to_numpy()
        print(f"  realised: MAE {np.abs(err).mean():.1f}, RMSE {np.sqrt((err ** 2).mean()):.1f} EUR/MWh")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
