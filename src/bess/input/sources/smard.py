"""SMARD price source (Bundesnetzagentur chart_data API): download of weekly chunks and parsing.

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

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

from bess.core import calendar as cal
from bess.input.prices import PriceSeries
from bess.input.sources.base import PriceSource, register_source


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


def download_raw(spec, cache_dir: Path, resolution: str, refresh_last: int = 2,
                 pause_s: float = 0.2) -> list[Path]:
    """Download all weekly chunks covering [spec.download_start, today] into cache_dir/raw/<res>/.

    Already cached chunks are kept except the `refresh_last` most recent ones,
    which may still be incomplete and are re-fetched.
    """
    d = spec.source
    base = f"{d['base_url']}/{d['filter_id']}/{d['region']}"
    index = _get_json(f"{base}/index_{resolution}.json")["timestamps"]
    start_ms = cal.to_ms(cal.day_start_utc(spec.download_start)) - 7 * 24 * cal.MS_PER_HOUR
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


@register_source("smard")
class SmardSource(PriceSource):
    """`source:` keys: base_url, filter_id, region (see configs/markets.yaml)."""

    resolutions = {"hour": 1.0, "quarterhour": 0.25}

    def fetch(self, spec, cache_dir: Path, resolution: str) -> PriceSeries:
        paths = download_raw(spec, cache_dir, resolution)
        return parse_raw(paths, self.resolutions[resolution])
