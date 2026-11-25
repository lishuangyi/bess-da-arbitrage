"""Download the day-ahead prices of a market (hour + quarterhour) and cache them as parquet.

Data: Bundesnetzagentur | SMARD.de (CC BY 4.0) for the markets declared with `source: {kind: smard}`; periods SMARD
leaves empty or has not published yet come from the market's `fallback` source (Energy-Charts, Fraunhofer ISE, same
SMARD data, CC BY 4.0) if both sources agree on the overlapping days. Raw data is not redistributed.

Usage: python -m bess download [--market de_lu | --config experiment.yaml] [--cache-dir DIR] [--no-fallback]
Writes <cache_dir>/prices_hour.parquet, prices_quarterhour.parquet and <results_dir>/data_report.json
(missing values, DST step counts, duplication checks, filled days). The market comes from --market, else from the
experiment config's `market:` key (default experiment.yaml -> de_lu).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from bess.core import calendar as cal
from bess.core.config import REPO_ROOT, experiment_config
from bess.input import prices
from bess.input.markets import MarketSpec, get_market
from bess.input.sources.base import get_source

OVERLAP_DAYS = 3          # complete primary days on each side of a gap used to check the fallback
MAX_ABS_DIFF = 0.011      # EUR/MWh; SMARD's hourly value is the mean of four rounded quarter-hours (0.005)


def _ranges(days: list[dt.date], split: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Contiguous runs of days, never crossing the first 15-min delivery day."""
    out = []
    for d in sorted(days):
        if out and d == out[-1][1] + dt.timedelta(days=1) and not (out[-1][1] < split <= d):
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


def fill_gaps(spec: MarketSpec, s: prices.PriceSeries, res: str, cache: Path, today: dt.date) -> tuple:
    """Fill the empty periods of `s` (incomplete days and days after its end up to tomorrow) from the fallback.
    Returns (series, report). Refuses (RuntimeError) if the sources disagree on the overlapping days."""
    rep = {"source": spec.fallback["kind"], "filled_days": [], "n_filled": 0, "overlap_n": 0,
           "overlap_max_abs_diff": None}
    first, last = cal.local_date(s.first_ts), cal.local_date(s.last_ts)
    gaps = [d for d in cal.daterange(first, last) if not s.is_complete(d)]
    gaps += cal.daterange(last + dt.timedelta(days=1), today + dt.timedelta(days=1))
    if not gaps:
        return s, rep
    src, qh = get_source(spec.fallback["kind"]), spec.qh_start_date
    diffs = []
    for a, b in _ranges(gaps, qh):
        lo = max(a - dt.timedelta(days=OVERLAP_DAYS), first)          # overlap days on the same side of the switch
        lo = min(max(lo, qh), a) if a >= qh else lo
        hi = max(b, min(b + dt.timedelta(days=OVERLAP_DAYS), last))
        hi = max(min(hi, qh - dt.timedelta(days=1)), b) if b < qh else hi
        fb = src.fetch_days(spec.fallback, lo, hi, cache, res)
        both = [(ts, fb.price[i]) for i, ts in enumerate(fb.timestamps_ms())]
        own_ms, own = s.timestamps_ms(), s.price
        idx = {int(t): v for t, v in zip(own_ms, own) if not np.isnan(v)}
        d = [abs(idx[int(t)] - v) for t, v in both if int(t) in idx and not np.isnan(v)]
        diffs += d
        if d and max(d) > MAX_ABS_DIFF:
            raise RuntimeError(f"{spec.id} {res}: fallback {spec.fallback['kind']} differs from the primary source by up "
                               f"to {max(d):.3f} EUR/MWh on {lo}..{hi}; not filling")
        if not d:
            raise RuntimeError(f"{spec.id} {res}: no overlapping periods to check the fallback around {a}..{b}")
        keep = np.zeros(len(fb.price), dtype=bool)
        for day in cal.daterange(a, b):
            if fb.is_complete(day):          # only whole days; partial days stay missing
                i0 = (cal.to_ms(cal.day_start_utc(day)) - fb.t0_ms) // fb.step_ms
                keep[i0:i0 + cal.expected_steps(day, fb.dt_h)] = True
        fb = prices.PriceSeries(fb.t0_ms, np.where(keep, fb.price, np.nan), fb.dt_h)
        s, filled = s.fill_from(fb)
        if len(filled):
            days = sorted({str(cal.local_date(cal.from_ms(int(t)))) for t in filled})
            rep["filled_days"] += days
            rep["n_filled"] += int(len(filled))
    rep["overlap_n"] = len(diffs)
    rep["overlap_max_abs_diff"] = float(max(diffs)) if diffs else None
    return s, rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--market", default=None, help="market id from configs/markets.yaml (overrides --config)")
    ap.add_argument("--config", default="experiment.yaml",
                    help="experiment config whose market and results_dir are used (e.g. experiment_at.yaml)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-fallback", action="store_true", help="primary source only (gaps stay missing)")
    args = ap.parse_args(argv)
    cfg = experiment_config(args.config)
    spec = get_market(args.market or cfg["market"])
    cache = Path(args.cache_dir) if args.cache_dir else spec.cache_path
    cache.mkdir(parents=True, exist_ok=True)
    source = get_source(spec.source["kind"])
    today = cal.local_date(dt.datetime.now(dt.timezone.utc))

    series, filled = {}, {}
    for res in ("hour", "quarterhour"):
        s = source.fetch(spec, cache, res).starting_at(cal.day_start_utc(spec.download_start))
        if spec.fallback and not args.no_fallback:
            s, filled[res] = fill_gaps(spec, s, res, cache, today)
        out = prices.cache_paths(cache)[res]
        s.to_frame().write_parquet(out)
        series[res] = s
        print(f"{spec.id} {res}: {len(s.price)} rows, {int(np.isnan(s.price).sum())} missing, "
              f"{s.first_ts} .. {s.last_ts} -> {out}")

    rep = prices.quality_report(series["hour"], series["quarterhour"], spec.qh_market_start)
    if filled:
        rep["filled_from_fallback"] = filled
    if args.market and args.market != cfg["market"]:
        res_dir = cache   # a market other than the config's: keep its report next to its data
    else:
        res_dir = REPO_ROOT / cfg["backtest"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "data_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        if isinstance(v, dict) and "incomplete_days" in v:
            v = {**v, "incomplete_days": f"{len(v['incomplete_days'])} days: {v['incomplete_days'][:10]}"}
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
