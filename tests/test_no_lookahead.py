"""No future information: decisions for delivery day D only use data published by D-1 12:00."""
import datetime as dt

import numpy as np
import pytest

from bess.backtest.engine import run_backtest
from bess.backtest.policy import Plan, Policy
from bess.data import calendar as cal
from bess.data.smard import MarketData, PriceSeries
from bess.forecast.base import check_information
from bess.forecast.naive import SeasonalNaive
from bess.markets.day_ahead import DayAheadMarket
from bess.optim.milp import ForecastMILP, PerfectForesight, solve_schedule

DAYS = [dt.date(2025, 3, 30), dt.date(2025, 7, 1), dt.date(2025, 10, 26), dt.date(2026, 1, 5)]


def perturbed_from(data: MarketData, day, value=1e4) -> MarketData:
    """Copy of `data` with every price at or after delivery day `day` replaced."""
    out = []
    cut = cal.to_ms(cal.day_start_utc(day))
    for s in (data.hour, data.quarterhour):
        p = s.price.copy()
        p[s.timestamps_ms() >= cut] = value
        out.append(PriceSeries(s.t0_ms, p, s.dt_h))
    return MarketData(*out)


@pytest.mark.parametrize("day", DAYS)
def test_information_set_at_gate_closure(mdata, day):
    info = mdata.known_at(cal.gate_closure_utc(day))
    for s in (info.hour, info.quarterhour):
        if s.empty:
            continue
        assert s.last_ts < cal.day_start_utc(day)
    # D-1 is fully known (published on D-2)
    assert info.hour.is_complete(day - dt.timedelta(days=1))
    # D becomes known only once published (13:00 on D-1), i.e. after gate closure
    later = mdata.known_at(cal.publication_utc(day))
    assert later.hour.is_complete(day)
    just_before = mdata.known_at(cal.publication_utc(day) - dt.timedelta(minutes=1))
    assert not just_before.hour.is_complete(day)
    assert cal.gate_closure_utc(day) < cal.publication_utc(day)


def test_check_information_raises(mdata):
    day = dt.date(2025, 7, 1)
    with pytest.raises(RuntimeError):
        check_information(day, mdata)
    check_information(day, mdata.known_at(cal.gate_closure_utc(day)))


@pytest.mark.parametrize("day", DAYS)
def test_naive_forecast_invariant_to_future(mdata, day):
    res = "quarterhour" if day >= dt.date(2025, 10, 1) else "hour"
    f = SeasonalNaive()
    a = f.predict(day, mdata.known_at(cal.gate_closure_utc(day)), res)
    pert = perturbed_from(mdata, day)
    b = f.predict(day, pert.known_at(cal.gate_closure_utc(day)), res)
    assert np.array_equal(a, b)
    assert len(a) == cal.expected_steps(day, cal.RESOLUTION_HOURS[res])
    assert np.all(a < 1e4)


class SpyPolicy(Policy):
    name = "spy"

    def __init__(self):
        self.seen = []

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None):
        assert oracle_prices is None
        self.seen.append((day, info.latest_timestamp(), info.as_of_utc))
        n = cal.expected_steps(day, cal.RESOLUTION_HOURS[resolution])
        r = solve_schedule(np.zeros(n), cal.RESOLUTION_HOURS[resolution], battery, [market], e_init=e_init)
        return Plan(r.schedule, r.solve_time_s, battery.e_final)


def test_engine_passes_only_gate_closure_information(mdata, battery):
    spy = SpyPolicy()
    days = [dt.date(2026, 1, 3), dt.date(2026, 1, 4), dt.date(2026, 1, 5)]
    run_backtest(spy, mdata, days, "quarterhour", battery, DayAheadMarket())
    for day, latest, as_of in spy.seen:
        assert as_of == cal.gate_closure_utc(day)
        assert latest < cal.day_start_utc(day)


def test_forecast_schedule_invariant_to_realised_prices(mdata, battery):
    """Changing the realised prices of D (and later) must change settlement, not the schedule."""
    days = [dt.date(2026, 1, 10)]
    pol = ForecastMILP(SeasonalNaive())
    a = run_backtest(pol, mdata, days, "quarterhour", battery, DayAheadMarket())
    pert = perturbed_from(mdata, days[0], value=500.0)
    b = run_backtest(pol, pert, days, "quarterhour", battery, DayAheadMarket())
    for col in ("discharge_mwh", "charge_mwh", "e_end_mwh"):
        assert a[col][0] == pytest.approx(b[col][0], abs=1e-9)
    assert a["revenue_eur"][0] != pytest.approx(b["revenue_eur"][0])


def test_oracle_only_for_perfect_foresight(mdata, battery):
    assert PerfectForesight.uses_oracle and not ForecastMILP.uses_oracle
    d = run_backtest(PerfectForesight(), mdata, [dt.date(2026, 1, 10)], "quarterhour", battery, DayAheadMarket())
    assert d["violations"][0] == 0


# ---------------------------------------------------------------- M2: XGB and rolling MILP

@pytest.fixture(scope="module")
def small_xgb(mdata):
    from bess.forecast import xgb as xgbmod
    from bess.forecast.xgb import XGBForecaster

    old = dict(xgbmod.XGB_PARAMS)
    xgbmod.XGB_PARAMS.update(n_estimators=30, early_stopping_rounds=5)
    try:
        train = cal.daterange("2025-03-01", "2025-07-31")
        val = cal.daterange("2025-08-01", "2025-09-15")
        yield XGBForecaster(horizons=(1, 2)).fit(mdata, train, val)
    finally:
        xgbmod.XGB_PARAMS.clear()
        xgbmod.XGB_PARAMS.update(old)


@pytest.mark.parametrize("day", [dt.date(2025, 9, 20), dt.date(2025, 10, 26), dt.date(2026, 1, 10)])
def test_xgb_forecast_invariant_to_future(mdata, small_xgb, day):
    res = "quarterhour" if day >= dt.date(2025, 10, 1) else "hour"
    t = cal.gate_closure_utc(day)
    a = small_xgb.predict(day, mdata.known_at(t), res)
    b = small_xgb.predict(day, perturbed_from(mdata, day).known_at(t), res)
    assert np.array_equal(a, b) and len(a) == cal.expected_steps(day, cal.RESOLUTION_HOURS[res])
    # horizon 2 (the day after D, used by the rolling MILP) with the same information set
    nxt = day + dt.timedelta(days=1)
    a2 = small_xgb.predict(nxt, mdata.known_at(t), res)
    b2 = small_xgb.predict(nxt, perturbed_from(mdata, day).known_at(t), res)
    assert np.array_equal(a2, b2)


def test_xgb_training_does_not_use_days_after_train_end(mdata):
    """Changing prices after the training window must not change the fitted model's
    training data (validation is used for early stopping only)."""
    from bess.forecast.xgb import dataset

    train = cal.daterange("2025-03-01", "2025-04-30")
    cut = cal.day_end_utc(max(train))
    X1, y1 = dataset(mdata.hour.truncated_before(cut), train, 1)
    X2, y2 = dataset(perturbed_from(mdata, dt.date(2025, 5, 1)).hour.truncated_before(cut), train, 1)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_rolling_schedule_invariant_to_realised_prices(mdata, battery, small_xgb):
    from bess.optim.rolling import RollingMILP

    days = [dt.date(2026, 1, 10), dt.date(2026, 1, 11)]
    pol = RollingMILP(small_xgb)
    a = run_backtest(pol, mdata, days, "quarterhour", battery, DayAheadMarket())
    # perturb D+1 (= second day) onwards: the 48 h plan for D sees only forecasts of D+1,
    # never its realised prices; the plan for D+1 sees only data up to D.
    b = run_backtest(pol, perturbed_from(mdata, days[1], value=500.0), days, "quarterhour", battery,
                     DayAheadMarket())
    for col in ("discharge_mwh", "charge_mwh", "e_end_mwh"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), atol=1e-9)
    assert (a["violations"] == 0).all()


def test_days_with_missing_gate_closure_inputs_are_excluded(mdata):
    """A day whose D-1 prices are missing is excluded for all methods (no interpolation)."""
    from bess.backtest.periods import period_days

    gap = dt.date(2026, 1, 9)
    out = []
    for s in (mdata.hour, mdata.quarterhour):
        p, ts = s.price.copy(), s.timestamps_ms()
        p[(ts >= cal.to_ms(cal.day_start_utc(gap))) & (ts < cal.to_ms(cal.day_end_utc(gap)))] = np.nan
        out.append(PriceSeries(s.t0_ms, p, s.dt_h))
    data = MarketData(*out)
    cfg = {"splits": {"test": {"start": "2026-01-05", "end": "2026-01-19", "resolution": "quarterhour"}}}
    days, _, info = period_days(cfg, data, "test")
    excluded = {gap, gap + dt.timedelta(days=1), gap + dt.timedelta(days=2), gap + dt.timedelta(days=7)}
    assert set(info["excluded_days"]) == {str(d) for d in excluded}
    assert info["n_incomplete_excluded"] == 1 and info["n_missing_inputs_excluded"] == 3
    assert not set(days) & excluded


def test_multi_segment_split(mdata):
    """A training split made of an hourly and a 15-min segment; single-segment access is refused."""
    from bess.backtest.periods import all_split_days, period_days, split_segments

    cfg = {"splits": {"train": [{"start": "2025-08-01", "end": "2025-09-30", "resolution": "hour"},
                                {"start": "2025-10-01", "end": "2025-11-30", "resolution": "quarterhour"}],
                      "validation": {"start": "2025-12-01", "end": "2025-12-31", "resolution": "quarterhour"}}}
    segs = split_segments(cfg, mdata, "train")
    assert [r for _, r, _ in segs] == ["hour", "quarterhour"]
    assert [len(d) for d, _, _ in segs] == [61, 61]
    days = all_split_days(cfg, mdata, "train")
    assert days == sorted(days) and len(days) == 122 and max(days) < dt.date(2025, 12, 1)
    with pytest.raises(ValueError):
        period_days(cfg, mdata, "train")
    v, res, _ = period_days(cfg, mdata, "validation")
    assert res == "quarterhour" and len(v) == 31


# ---------------------------------------------------------------- forward-chaining RL training forecasts

def test_forward_fold_forecasts_use_only_earlier_years(mdata):
    """A day of year y is forecast by a model fitted on earlier years only; lags stay <= T-1.
    Perturbing prices from day X on must leave every forecast for days <= X unchanged, and the
    first year (no earlier history) gets no forecasts."""
    from bess.forecast import xgb as xgbmod

    old = dict(xgbmod.XGB_PARAMS)
    xgbmod.XGB_PARAMS.update(n_estimators=30, early_stopping_rounds=5)
    try:
        days = cal.daterange("2025-06-01", "2026-01-18")
        cut = dt.date(2026, 1, 9)
        a = xgbmod.forward_fold_forecasts(mdata, days, k=1)
        b = xgbmod.forward_fold_forecasts(perturbed_from(mdata, cut), days, k=1)
    finally:
        xgbmod.XGB_PARAMS.clear()
        xgbmod.XGB_PARAMS.update(old)
    assert all(d.year == 2026 for d in a)                 # 2025 has no earlier history in the data
    assert sorted(a) == [d for d in days if d.year == 2026]
    for d in a:
        if d <= cut:
            assert np.array_equal(a[d], b[d]), d
    assert any(not np.array_equal(a[d], b[d]) for d in a if d > cut)
