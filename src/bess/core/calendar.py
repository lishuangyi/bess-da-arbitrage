"""Delivery-day calendar in Europe/Berlin.

Conventions: scalar instants are timezone-aware `datetime` objects in UTC; arrays of
instants are `numpy.datetime64[ms]` holding UTC (naive) values; vectorised time-zone
conversion uses polars.

A *delivery day* D is the local calendar day in Europe/Berlin; it spans
[local 00:00 of D, local 00:00 of D+1), which is 23, 24 or 25 hours long (DST
transitions). At 60-min resolution this gives 23/24/25 steps, at 15-min
resolution 92/96/100 steps.

Information timing for the day-ahead (DA) auction of delivery day D:
- gate closure: D-1 12:00 local;
- results (prices for D) are published shortly after; we use a conservative
  D-1 13:00 local as publication time (configurable).
A price of delivery day X is therefore known at time t iff publication(X) <= t.
At gate closure of D, every delivery day <= D-1 is known and D is not.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

TZ = "Europe/Berlin"
ZONE = ZoneInfo(TZ)
UTC = dt.timezone.utc
MS_PER_HOUR = 3_600_000

RESOLUTION_HOURS = {"hour": 1.0, "quarterhour": 0.25}


def to_date(d) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[D]").item()
    return dt.date.fromisoformat(str(d)[:10])


def to_ms(t: dt.datetime) -> int:
    """UTC epoch milliseconds of an aware datetime."""
    if t.tzinfo is None:
        raise ValueError("naive datetime")
    delta = t - dt.datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000


def from_ms(ms: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1, tzinfo=UTC) + dt.timedelta(milliseconds=int(ms))


def day_start_utc(day) -> dt.datetime:
    """UTC instant of local midnight at the start of delivery day `day`."""
    return dt.datetime.combine(to_date(day), dt.time(0), tzinfo=ZONE).astimezone(UTC)


def day_end_utc(day) -> dt.datetime:
    """UTC instant of local midnight at the end of delivery day `day` (exclusive)."""
    return day_start_utc(to_date(day) + dt.timedelta(days=1))


def day_hours(day) -> float:
    return (day_end_utc(day) - day_start_utc(day)).total_seconds() / 3600.0


def expected_steps(day, dt_h: float) -> int:
    n = day_hours(day) / dt_h
    assert abs(n - round(n)) < 1e-9
    return int(round(n))


@lru_cache(maxsize=16384)
def _day_index(day: dt.date, dt_h: float) -> np.ndarray:
    start = np.datetime64(to_ms(day_start_utc(day)), "ms")
    step = np.timedelta64(int(round(dt_h * MS_PER_HOUR)), "ms")
    out = start + np.arange(expected_steps(day, dt_h)) * step
    out.setflags(write=False)
    return out


def day_index(day, dt_h: float) -> np.ndarray:
    """UTC start instants (datetime64[ms]) of all delivery periods of `day`.

    Depends only on the calendar, never on price data.
    """
    return _day_index(to_date(day), float(dt_h))


def _local(ts) -> pl.Series:
    arr = np.asarray(ts).astype("datetime64[ms]")
    return pl.Series("ts", arr).dt.replace_time_zone("UTC").dt.convert_time_zone(TZ)


def local_date_of(ts) -> np.ndarray:
    """Local delivery day (datetime.date) of each UTC instant (datetime64 array)."""
    return np.array(_local(ts).dt.date().to_list(), dtype=object)


def local_date(t: dt.datetime) -> dt.date:
    """Local delivery day of one aware instant."""
    return t.astimezone(ZONE).date()


@lru_cache(maxsize=16384)
def _day_minutes(day: dt.date, dt_h: float) -> np.ndarray:
    out = local_minute_of_day(_day_index(day, dt_h))
    out.setflags(write=False)
    return out


def local_minute_of_day(ts) -> np.ndarray:
    """Local wall-clock minute of day (0..1439) of each UTC instant.

    On the autumn DST day the 02:00-03:00 hour occurs twice and maps to the same
    wall-clock minutes; on the spring DST day 02:00-03:00 does not occur.
    """
    s = _local(ts)
    return (s.dt.hour().cast(pl.Int64) * 60 + s.dt.minute().cast(pl.Int64)).to_numpy()


def day_local_minutes(day, dt_h: float) -> np.ndarray:
    """Local start minute of every period of delivery day `day` (cached)."""
    return _day_minutes(to_date(day), float(dt_h))


def daterange(start, end) -> list[dt.date]:
    s, e = to_date(start), to_date(end)
    return [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]


def _local_time(day: dt.date, hhmm: str) -> dt.datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return dt.datetime.combine(day, dt.time(h, m), tzinfo=ZONE).astimezone(UTC)


def gate_closure_utc(delivery_day, hhmm: str = "12:00") -> dt.datetime:
    """DA gate closure for delivery day D: D-1 at `hhmm` local time."""
    return _local_time(to_date(delivery_day) - dt.timedelta(days=1), hhmm)


def publication_utc(delivery_day, hhmm: str = "13:00") -> dt.datetime:
    """(Conservative) publication time of the DA prices of delivery day D: D-1 `hhmm` local."""
    return _local_time(to_date(delivery_day) - dt.timedelta(days=1), hhmm)
