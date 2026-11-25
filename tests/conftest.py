import numpy as np
import pytest

from bess.core import calendar as cal
from bess.core.config import Battery
from bess.input.prices import MarketData, PriceSeries


def synthetic_market_data(start="2025-02-01", end="2026-01-20", qh_start="2025-10-01", seed=0) -> MarketData:
    rng = np.random.default_rng(seed)
    s_ms, e_ms = cal.to_ms(cal.day_start_utc(start)), cal.to_ms(cal.day_end_utc(end))
    step = cal.MS_PER_HOUR // 4
    ts = np.arange(s_ms, e_ms, step, dtype=np.int64)
    h = (ts // cal.MS_PER_HOUR) % 24 + ((ts // step) % 4) / 4
    qp = 80 + 40 * np.sin(2 * np.pi * (h - 8) / 24) + rng.normal(0, 25, len(ts))
    qp[rng.random(len(ts)) < 0.03] -= 150  # some negative prices
    hour = PriceSeries(s_ms, qp.reshape(-1, 4).mean(axis=1), 1.0)
    q = PriceSeries(s_ms, qp, 0.25).starting_at(cal.day_start_utc(qh_start))
    return MarketData(hour, q)


@pytest.fixture(scope="session")
def mdata() -> MarketData:
    return synthetic_market_data()


@pytest.fixture(scope="session")
def battery() -> Battery:
    return Battery.from_yaml()
