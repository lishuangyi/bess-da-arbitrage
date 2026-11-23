"""Delivery-day splitting in Europe/Berlin, incl. DST days (23/25 h, 92/100 quarter-hours)."""
import datetime as dt

import numpy as np
import pytest

from bess.config import REPO_ROOT, experiment_config
from bess.data import calendar as cal
from bess.data.smard import PriceSeries, load_market_data
from bess.forecast.base import profile_on_grid

SPRING = ["2020-03-29", "2021-03-28", "2022-03-27", "2023-03-26", "2024-03-31", "2025-03-30", "2026-03-29"]
AUTUMN = ["2020-10-25", "2021-10-31", "2022-10-30", "2023-10-29", "2024-10-27", "2025-10-26", "2026-10-25"]


@pytest.mark.parametrize("day", SPRING)
def test_spring_forward_steps(day):
    assert cal.expected_steps(day, 1.0) == 23
    assert cal.expected_steps(day, 0.25) == 92


@pytest.mark.parametrize("day", AUTUMN)
def test_fall_back_steps(day):
    assert cal.expected_steps(day, 1.0) == 25
    assert cal.expected_steps(day, 0.25) == 100


def test_normal_day_and_boundaries():
    assert cal.expected_steps("2025-06-15", 1.0) == 24
    assert cal.expected_steps("2025-06-15", 0.25) == 96
    # CET: local midnight = 23:00 UTC previous day; CEST: 22:00 UTC
    utc = dt.timezone.utc
    assert cal.day_start_utc("2025-03-30") == dt.datetime(2025, 3, 29, 23, tzinfo=utc)
    assert cal.day_end_utc("2025-03-30") == dt.datetime(2025, 3, 30, 22, tzinfo=utc)
    assert cal.day_start_utc("2025-10-26") == dt.datetime(2025, 10, 25, 22, tzinfo=utc)
    assert cal.day_end_utc("2025-10-26") == dt.datetime(2025, 10, 26, 23, tzinfo=utc)


def test_year_partition():
    for year, hours in ((2024, 8784), (2025, 8760)):
        days = cal.daterange(f"{year}-01-01", f"{year}-12-31")
        assert sum(cal.expected_steps(d, 1.0) for d in days) == hours
        idx = np.concatenate([cal.day_index(d, 0.25) for d in days])
        assert len(np.unique(idx)) == len(idx) == hours * 4 and np.all(np.diff(idx).astype(np.int64) == 900_000)
        assert (cal.local_date_of(idx[[0, -1]]) == [dt.date(year, 1, 1), dt.date(year, 12, 31)]).all()


def test_day_prices_lengths(mdata):
    for day, nh in (("2025-03-30", 23), ("2025-10-26", 25), ("2025-07-01", 24)):
        assert len(mdata.hour.day_prices(day)) == nh
    assert len(mdata.quarterhour.day_prices("2025-10-26")) == 100
    assert np.isfinite(mdata.quarterhour.day_prices("2025-10-26")).all()


def _local_hour_series(start, end) -> PriceSeries:
    """Hourly series whose price equals the local wall-clock hour."""
    t0 = cal.to_ms(cal.day_start_utc(start))
    ts = np.arange(t0, cal.to_ms(cal.day_end_utc(end)), cal.MS_PER_HOUR, dtype=np.int64)
    return PriceSeries(t0, (cal.local_minute_of_day(ts.astype("datetime64[ms]")) // 60).astype(float), 1.0)


def test_profile_mapping_across_resolution_and_dst():
    s = _local_hour_series("2025-03-23", "2025-03-31")
    # hourly source -> 15-min target: forward fill within the hour
    prof = profile_on_grid(s, dt.date(2025, 3, 23), dt.date(2025, 3, 24), 0.25)
    assert len(prof) == 96 and (prof == np.repeat(np.arange(24.0), 4)).all()
    # spring DST source (no 02:00) -> normal target: 02:00 taken from 01:00
    prof = profile_on_grid(s, dt.date(2025, 3, 30), dt.date(2025, 3, 31), 1.0)
    assert len(prof) == 24 and prof[2] == 1.0 and prof[3] == 3.0
    # normal source -> spring DST target: 23 values, 02:00 skipped
    prof = profile_on_grid(s, dt.date(2025, 3, 23), dt.date(2025, 3, 30), 1.0)
    assert len(prof) == 23 and prof[2] == 3.0


def test_profile_autumn_target():
    s = _local_hour_series("2025-10-19", "2025-10-26")
    prof = profile_on_grid(s, dt.date(2025, 10, 19), dt.date(2025, 10, 26), 1.0)
    assert len(prof) == 25 and list(prof[:5]) == [0.0, 1.0, 2.0, 2.0, 3.0]


@pytest.mark.skipif(not (REPO_ROOT / "data" / "prices_hour.parquet").exists(), reason="data not downloaded")
def test_real_data_dst_counts():
    data = load_market_data(experiment_config())
    for series, dt_h in ((data.hour, 1.0), (data.quarterhour, 0.25)):
        for day in SPRING + AUTUMN:
            d = cal.to_date(day)
            if series.days() and series.days()[0] < d <= series.days()[-1] and series.is_complete(d):
                assert len(series.day_prices(d)) == cal.expected_steps(d, dt_h)
    assert len(data.hour.day_prices("2024-03-31")) == 23 and data.hour.is_complete("2024-03-31")
    assert len(data.hour.day_prices("2024-10-27")) == 25 and data.hour.is_complete("2024-10-27")
    assert data.quarterhour.is_complete("2025-10-26") and len(data.quarterhour.day_prices("2025-10-26")) == 100
    assert data.quarterhour.is_complete("2026-03-29") and len(data.quarterhour.day_prices("2026-03-29")) == 92
