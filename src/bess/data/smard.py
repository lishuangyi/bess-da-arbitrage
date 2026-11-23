"""SMARD day-ahead price download, caching and access.

Data: Bundesnetzagentur | SMARD.de (CC BY 4.0).

API (chart_data, see github.com/bundesAPI/smard-api):
  {base}/{filter}/{region}/index_{res}.json          -> {"timestamps": [ms, ...]} (weekly chunks)
  {base}/{filter}/{region}/{filter}_{region}_{res}_{ts}.json -> {"series": [[ms, value|null], ...]}
Timestamps are UTC epoch milliseconds marking the start of each delivery period.

Verified on 2026-11-21 for filter 4169 / DE-LU:
- `quarterhour` values before 2025-10-01 are the hourly price repeated 4 times
  (not genuine 15-min prices), so the 15-min series is only used from the
  first native 15-min auction (delivery day 2025-10-01) on;
- from 2025-10-01 the `hour` series equals the mean of the four quarter-hour prices.
The download script re-checks both facts and writes them to the data report.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import requests

from bess.data import calendar as cal

RESOLUTIONS = ("hour", "quarterhour")


# --------------------------------------------------------------------------- download

def _get_json(url: str, retries: int = 4, timeout: float = 30.0) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "bess-da-arbitrage/0.1"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:  # pragma: no cover - network
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def download_raw(cfg: dict, cache_dir: Path, resolution: str, refresh_last: int = 2,
                 pause_s: float = 0.2) -> list[Path]:
    """Download all weekly chunks covering [download_start, today] into cache_dir/raw/<res>/.

    Already cached chunks are kept except the `refresh_last` most recent ones,
    which may still be incomplete and are re-fetched.
    """
    d = cfg["data"]
    base = f"{d['smard_base_url']}/{d['filter_id']}/{d['region']}"
    index = _get_json(f"{base}/index_{resolution}.json")["timestamps"]
    start_ms = cal.to_ms(cal.day_start_utc(d["download_start"])) - 7 * 24 * cal.MS_PER_HOUR
    chunks = sorted(t for t in index if t >= start_ms)
    out_dir = cache_dir / "raw" / resolution
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ts in enumerate(chunks):
        p = out_dir / f"{ts}.json"
        if not p.exists() or i >= len(chunks) - refresh_last:
            js = _get_json(f"{base}/{d['filter_id']}_{d['region']}_{resolution}_{ts}.json")
            p.write_text(json.dumps(js))
            time.sleep(pause_s)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- price series

@dataclass
class PriceSeries:
    """Prices on a regular UTC grid: period k starts at t0_ms + k * step. NaN = missing."""

    t0_ms: int
    price: np.ndarray
    dt_h: float

    def __post_init__(self):
        self.price = np.asarray(self.price, dtype=float)
        self.t0_ms = int(self.t0_ms)

    # ---- construction / conversion
    @property
    def step_ms(self) -> int:
        return int(round(self.dt_h * cal.MS_PER_HOUR))

    @classmethod
    def from_points(cls, ms: np.ndarray, price: np.ndarray, dt_h: float) -> "PriceSeries":
        """Place (timestamp, price) points on a regular grid from first to last point."""
        ms = np.asarray(ms, dtype=np.int64)
        price = np.asarray(price, dtype=float)
        if len(ms) == 0:
            return cls(0, np.empty(0), dt_h)
        step = int(round(dt_h * cal.MS_PER_HOUR))
        off = ms - ms.min()
        if np.any(off % step):
            raise ValueError("timestamps are not on a regular grid")
        k = off // step
        grid = np.full(int(k.max()) + 1, np.nan)
        grid[k] = price
        return cls(int(ms.min()), grid, dt_h)

    @classmethod
    def from_frame(cls, df: pl.DataFrame, dt_h: float) -> "PriceSeries":
        ms = df["ts_utc"].dt.epoch("ms").to_numpy()
        return cls.from_points(ms, df["price"].to_numpy(), dt_h)

    def timestamps_ms(self) -> np.ndarray:
        return self.t0_ms + np.arange(len(self.price), dtype=np.int64) * self.step_ms

    def to_frame(self) -> pl.DataFrame:
        ts = pl.Series("ts_utc", self.timestamps_ms()).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        return pl.DataFrame({"ts_utc": ts, "price": self.price})

    # ---- basic info
    @property
    def empty(self) -> bool:
        return len(self.price) == 0

    @property
    def resolution(self) -> str:
        return "hour" if self.dt_h == 1.0 else "quarterhour"

    @property
    def first_ts(self) -> dt.datetime | None:
        return None if self.empty else cal.from_ms(self.t0_ms)

    @property
    def last_ts(self) -> dt.datetime | None:
        return None if self.empty else cal.from_ms(self.t0_ms + (len(self.price) - 1) * self.step_ms)

    # ---- day access
    def day_prices(self, day) -> np.ndarray:
        """Prices of delivery day `day` on the calendar grid (NaN where missing)."""
        n = cal.expected_steps(day, self.dt_h)
        out = np.full(n, np.nan)
        if self.empty:
            return out
        off = cal.to_ms(cal.day_start_utc(day)) - self.t0_ms
        if off % self.step_ms:
            raise ValueError("day start not on the price grid")
        i0 = off // self.step_ms
        lo, hi = max(i0, 0), min(i0 + n, len(self.price))
        if lo < hi:
            out[lo - i0:hi - i0] = self.price[lo:hi]
        return out

    def is_complete(self, day) -> bool:
        return bool(np.isfinite(self.day_prices(day)).all())

    def days(self) -> list:
        if self.empty:
            return []
        return cal.daterange(cal.local_date(self.first_ts), cal.local_date(self.last_ts))

    def complete_days(self, start=None, end=None) -> list:
        ds = self.days()
        if start is not None:
            ds = [d for d in ds if d >= cal.to_date(start)]
        if end is not None:
            ds = [d for d in ds if d <= cal.to_date(end)]
        return [d for d in ds if self.is_complete(d)]

    # ---- slicing
    def _n_before(self, t: dt.datetime) -> int:
        k = -(-(cal.to_ms(t) - self.t0_ms) // self.step_ms)  # ceil
        return int(min(max(k, 0), len(self.price)))

    def truncated_before(self, t: dt.datetime) -> "PriceSeries":
        """Copy with only periods starting strictly before `t`."""
        return PriceSeries(self.t0_ms, self.price[: self._n_before(t)].copy(), self.dt_h)

    def starting_at(self, t: dt.datetime) -> "PriceSeries":
        """Copy with only periods starting at or after `t`."""
        k = self._n_before(t)
        return PriceSeries(self.t0_ms + k * self.step_ms, self.price[k:].copy(), self.dt_h)


def parse_raw(paths: list[Path], dt_h: float) -> PriceSeries:
    """Merge raw chunks onto a regular UTC grid; missing values stay NaN (no interpolation).

    Trailing nulls (future periods not yet published) are dropped.
    """
    rows = []
    for p in paths:
        rows.extend(json.loads(Path(p).read_text())["series"])
    df = (pl.DataFrame(rows, schema=["ms", "price"], orient="row")
          .with_columns(pl.col("ms").cast(pl.Int64), pl.col("price").cast(pl.Float64))
          .unique("ms", keep="last").sort("ms"))
    valid = df.filter(pl.col("price").is_not_null())
    if valid.is_empty():
        raise ValueError("no data")
    df = df.filter(pl.col("ms") <= valid["ms"].max())
    return PriceSeries.from_points(df["ms"].to_numpy(), df["price"].fill_null(np.nan).to_numpy(), dt_h)


def last_published_day(t: dt.datetime, publication_hhmm: str = "13:00") -> dt.date:
    """Latest delivery day X whose DA prices are published at or before time t."""
    x = cal.local_date(t) + dt.timedelta(days=1)
    while cal.publication_utc(x, publication_hhmm) > t:
        x = x - dt.timedelta(days=1)
    return x


@dataclass
class MarketData:
    """Hourly and native 15-min day-ahead prices.

    `hour`: hourly prices over the full range (from 2025-10-01 equal to the
    average of the four quarter-hour prices, as published by SMARD).
    `quarterhour`: native 15-min prices, only from the first 15-min auction on.
    """

    hour: PriceSeries
    quarterhour: PriceSeries
    as_of_utc: dt.datetime | None = None  # None = full data (only for settlement/evaluation)

    def series(self, resolution: str) -> PriceSeries:
        return self.hour if resolution == "hour" else self.quarterhour

    def known_at(self, t: dt.datetime, publication_hhmm: str = "13:00") -> "MarketData":
        """Information set at time t: only delivery days already published at t."""
        cut = cal.day_end_utc(last_published_day(t, publication_hhmm))
        return MarketData(self.hour.truncated_before(cut), self.quarterhour.truncated_before(cut), as_of_utc=t)

    def latest_timestamp(self) -> dt.datetime | None:
        ts = [s.last_ts for s in (self.hour, self.quarterhour) if not s.empty]
        return max(ts) if ts else None


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {r: Path(cache_dir) / f"prices_{r}.parquet" for r in RESOLUTIONS}


def load_market_data(cfg: dict, cache_dir: Path | None = None) -> MarketData:
    from bess.config import REPO_ROOT

    cache_dir = Path(cache_dir or REPO_ROOT / cfg["data"]["cache_dir"])
    paths = cache_paths(cache_dir)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run scripts/download_data.py first")
    hour = PriceSeries.from_frame(pl.read_parquet(paths["hour"]), 1.0)
    qh = PriceSeries.from_frame(pl.read_parquet(paths["quarterhour"]), 0.25)
    qh = qh.starting_at(cal.day_start_utc(cfg["data"]["qh_market_start"]))
    return MarketData(hour, qh)


# --------------------------------------------------------------------------- quality report

def quality_report(hour: PriceSeries, qh_all: PriceSeries, qh_start: str) -> dict:
    """Missing values, DST step counts and the duplication checks described in the module doc."""
    start = cal.day_start_utc(qh_start)
    rep: dict = {}
    for name, s in (("hour", hour), ("quarterhour", qh_all.starting_at(start))):
        days = s.days()[1:]  # first day may be partial due to chunk boundaries
        incomplete = [str(d) for d in days if not s.is_complete(d)]
        counts = Counter(cal.expected_steps(d, s.dt_h) for d in days)
        rep[name] = {
            "first_ts_utc": str(s.first_ts), "last_ts_utc": str(s.last_ts),
            "n_values": int(np.isfinite(s.price).sum()), "n_missing": int(np.isnan(s.price).sum()),
            "incomplete_days": incomplete,
            "steps_per_day_counts": {int(k): int(v) for k, v in sorted(counts.items())},
            "min_price": float(np.nanmin(s.price)), "max_price": float(np.nanmax(s.price)),
        }
    hour_df = hour.to_frame().rename({"price": "hour_price"})
    q = qh_all.to_frame().drop_nulls().filter(pl.col("price").is_not_nan())
    q = q.with_columns(pl.col("ts_utc").dt.truncate("1h").alias("h"))
    pre = q.filter(pl.col("ts_utc") < start)
    if pre.height:
        g = (pre.group_by("h").agg(pl.col("price").n_unique().alias("n_unique"), pl.col("price").first().alias("p"))
             .join(hour_df, left_on="h", right_on="ts_utc", how="left"))
        rep["pre_switch_quarterhour_check"] = {
            "n_hours": g.height,
            "share_constant_within_hour": float((g["n_unique"] == 1).mean()),
            "share_equal_to_hourly_price": float(((g["p"] - g["hour_price"]).abs() <= 1e-6).fill_null(False).mean()),
        }
    post = q.filter(pl.col("ts_utc") >= start)
    if post.height:
        g = (post.group_by("h").agg(pl.col("price").mean().alias("m"))
             .join(hour_df, left_on="h", right_on="ts_utc", how="inner")
             .with_columns((pl.col("hour_price") - pl.col("m")).abs().alias("d")).drop_nulls("d")
             .filter(pl.col("d").is_not_nan()))
        rep["post_switch_hour_vs_qh_mean"] = {
            "n_hours": g.height, "max_abs_diff": float(g["d"].max()),
            "share_within_0.01": float((g["d"] <= 0.01 + 1e-9).mean()),
        }
    return rep
