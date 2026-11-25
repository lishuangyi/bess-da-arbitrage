"""Gap filling of the price cache from a fallback source (no network: a fake source stands in)."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from bess.core import calendar as cal
from bess.input import download
from bess.input.markets import get_market
from bess.input.prices import PriceSeries
from bess.input.sources.base import PriceSource, register_source
from bess.input.sources.energy_charts import to_resolution


def _series(first, last, dt_h, seed=0, offset=0.0):
    ms0 = cal.to_ms(cal.day_start_utc(first))
    n = int(round((cal.to_ms(cal.day_end_utc(last)) - ms0) / (dt_h * cal.MS_PER_HOUR)))
    p = 80 + np.random.default_rng(seed).normal(0, 20, n) + offset
    return PriceSeries(ms0, np.round(p, 2), dt_h)


TRUTH = _series("2026-11-01", "2026-11-26", 0.25)


@register_source("fake_test")
class FakeSource(PriceSource):
    offset = 0.0

    def fetch(self, spec, cache_dir, resolution):
        raise NotImplementedError

    def fetch_days(self, selector, first, last, cache_dir, resolution):
        lo, hi = cal.to_ms(cal.day_start_utc(first)), cal.to_ms(cal.day_end_utc(last))
        ms = TRUTH.timestamps_ms()
        keep = (ms >= lo) & (ms < hi)
        return to_resolution(ms[keep], TRUTH.price[keep] + FakeSource.offset, TRUTH.step_ms, 0.25)


def test_fill_from_keeps_own_values_and_reports_filled_periods():
    a = PriceSeries(0, np.array([1.0, np.nan, 3.0]), 1.0)
    b = PriceSeries(cal.MS_PER_HOUR, np.array([9.0, 9.0, 4.0, 5.0]), 1.0)
    s, filled = a.fill_from(b)
    np.testing.assert_array_equal(s.price, [1.0, 9.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(filled, np.array([1, 3, 4]) * cal.MS_PER_HOUR)


def test_to_resolution_quarter_hours_to_hours_needs_complete_hours():
    ms = np.arange(8, dtype=np.int64) * (cal.MS_PER_HOUR // 4)
    s = to_resolution(ms[:7], np.arange(7.0), cal.MS_PER_HOUR // 4, 1.0)
    np.testing.assert_allclose(s.price[0], 1.5)
    assert np.isnan(s.price[1])


def test_fill_gaps_fills_only_missing_days_after_checking_the_overlap(tmp_path):
    spec = dataclasses.replace(get_market("de_lu"), fallback={"kind": "fake_test"})
    s = PriceSeries(TRUTH.t0_ms, TRUTH.price.copy(), 0.25)
    for d in ("2026-11-20", "2026-11-21"):
        i0 = (cal.to_ms(cal.day_start_utc(d)) - s.t0_ms) // s.step_ms
        s.price[i0:i0 + 96] = np.nan
    cut = cal.to_ms(cal.day_start_utc("2026-11-24"))
    s = PriceSeries(s.t0_ms, s.price[: (cut - s.t0_ms) // s.step_ms], 0.25)     # primary ends on 11-23
    FakeSource.offset = 0.0
    out, rep = download.fill_gaps(spec, s, "quarterhour", tmp_path, dt.date(2026, 11, 25))
    assert rep["filled_days"] == ["2026-11-20", "2026-11-21", "2026-11-24", "2026-11-25", "2026-11-26"]
    assert rep["overlap_max_abs_diff"] == 0.0 and rep["overlap_n"] > 0
    np.testing.assert_array_equal(out.price, TRUTH.price)
    FakeSource.offset = 0.5           # a fallback that disagrees is refused
    with pytest.raises(RuntimeError, match="differs"):
        download.fill_gaps(spec, s, "quarterhour", tmp_path, dt.date(2026, 11, 25))
    FakeSource.offset = 0.0
