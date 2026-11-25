"""Energy-Charts price API (Fraunhofer ISE), used as a gap-filling fallback for SMARD.

  GET https://api.energy-charts.info/price?bzn=<zone>&start=YYYY-MM-DD&end=YYYY-MM-DD
  -> {"unix_seconds": [...], "price": [...], "unit": "EUR / MWh", "license_info": "CC BY 4.0 ... from
      Bundesnetzagentur | SMARD.de"}
Timestamps are UTC seconds marking the start of each delivery period, at the market's native resolution (60 min before
the 15-min auction, 15 min after). The values are the same day-ahead auction prices SMARD publishes; the download
command only uses them for periods SMARD leaves empty, after checking that both sources agree on the overlapping days.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import requests

from bess.core import calendar as cal
from bess.input.prices import PriceSeries
from bess.input.sources.base import PriceSource, register_source

BASE_URL = "https://api.energy-charts.info/price"


def _get_json(params: dict, retries: int = 4, timeout: float = 60.0) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=timeout, headers={"User-Agent": "bess-da-arbitrage/0.1"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:  # pragma: no cover - network
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"failed to fetch {BASE_URL} {params}: {last}")


def to_resolution(ms: np.ndarray, price: np.ndarray, step_ms: int, dt_h: float) -> PriceSeries:
    """Native-resolution points -> series at dt_h. 15 min -> 60 min: mean of the four quarter-hours (SMARD's
    convention, verified to 0.005 EUR/MWh); 60 min -> 15 min: the hourly price repeated (SMARD's quarter-hour
    series before the 15-min auction)."""
    target = int(round(dt_h * cal.MS_PER_HOUR))
    if step_ms == target:
        return PriceSeries.from_points(ms, price, dt_h)
    if step_ms < target:
        k = target // step_ms
        start = ms - ms % target
        grid = np.unique(start)
        out = np.full(len(grid), np.nan)
        idx = np.searchsorted(grid, start)
        cnt = np.bincount(idx, minlength=len(grid))
        sums = np.bincount(idx, weights=price, minlength=len(grid))
        ok = cnt == k                      # only complete hours
        out[ok] = sums[ok] / k
        return PriceSeries.from_points(grid, out, dt_h)
    k = step_ms // target
    rep_ms = (ms[:, None] + np.arange(k)[None, :] * target).ravel()
    return PriceSeries.from_points(rep_ms, np.repeat(price, k), dt_h)


@register_source("energy_charts")
class EnergyChartsSource(PriceSource):
    """`source:` keys: bzn (bidding zone, e.g. DE-LU). Only day ranges are fetched (`fetch_days`)."""

    resolutions = {"hour": 1.0, "quarterhour": 0.25}

    def fetch(self, spec, cache_dir: Path, resolution: str):
        raise NotImplementedError("energy_charts is used as a fallback for day ranges only (fetch_days)")

    def fetch_days(self, selector: dict, first: dt.date, last: dt.date, cache_dir: Path, resolution: str) -> PriceSeries:
        """Prices of delivery days first..last (inclusive) at `resolution`; the raw answer is kept under
        cache_dir/raw/energy_charts/."""
        params = {"bzn": selector["bzn"], "start": str(first - dt.timedelta(days=1)),
                  "end": str(last + dt.timedelta(days=1))}
        js = _get_json(params)
        raw = Path(cache_dir) / "raw" / "energy_charts"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"{selector['bzn']}_{first}_{last}.json").write_text(json.dumps(js))
        sec = np.asarray(js["unix_seconds"], dtype=np.int64)
        price = np.asarray([np.nan if v is None else v for v in js["price"]], dtype=float)
        if len(sec) < 2:
            raise RuntimeError(f"energy_charts returned no data for {params}")
        ms = sec * 1000
        steps = np.unique(np.diff(ms))
        lo, hi = cal.to_ms(cal.day_start_utc(first)), cal.to_ms(cal.day_end_utc(last))
        if len(steps) != 1:
            raise NotImplementedError(f"energy_charts range {first}..{last} mixes resolutions {steps.tolist()}; "
                                      "fetch the days before and after the 15-min switch separately")
        keep = (ms >= lo) & (ms < hi)
        return to_resolution(ms[keep], price[keep], int(steps[0]), self.resolutions[resolution])
