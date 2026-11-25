"""Shared backtester: same timing rules and settlement for every method.

For each delivery day D:
1. information set = everything published by DA gate closure (D-1 12:00);
2. the policy returns a full-day schedule (DA positions are locked for the whole day);
3. the schedule is settled at the realised DA prices of D (settlement only);
4. physical constraints are re-checked from the powers.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl

from bess.core import calendar as cal
from bess.core.config import Battery
from bess.core.contracts import Policy
from bess.core.markets.base import Market
from bess.forecasting.base import check_information
from bess.input.prices import MarketData
from bess.milp.model import count_violations


def settle_day(policy: Policy, data: MarketData, day, resolution: str, e_init: float, battery: Battery,
               market: Market, publication_hhmm: str = "13:00") -> dict:
    """Plan `day` from start energy `e_init` with gate-closure information, settle at realised prices."""
    day = cal.to_date(day)
    dt_h = cal.RESOLUTION_HOURS[resolution]
    actual = data.series(resolution).day_prices(day)
    if not np.isfinite(actual).all():
        raise ValueError(f"{day}: incomplete realised prices - filter days with complete_days()")
    t_gc = market.gate_closure_utc(day)
    info = data.known_at(t_gc, publication_hhmm)
    check_information(day, info)
    t0 = time.perf_counter()
    plan = policy.plan(day, info, resolution, e_init, battery, market,
                       oracle_prices=actual.copy() if policy.uses_oracle else None)
    wall = time.perf_counter() - t0
    s = plan.schedule
    assert s.n == len(actual) == cal.expected_steps(day, dt_h), (day, s.n, len(actual))
    revenue = float(np.sum(market.settle(s, actual)))
    e_dis = float(np.sum(s.p_dis) * dt_h)
    deg = battery.c_deg_eur_per_mwh * e_dis
    return {
        "day": day, "n_steps": s.n, "revenue_eur": revenue, "deg_cost_eur": deg,
        "net_eur": revenue - deg, "discharge_mwh": e_dis, "charge_mwh": float(np.sum(s.p_ch) * dt_h),
        "efc": e_dis / battery.e_max_mwh,
        "violations": count_violations(s, battery, e_init, plan.e_final_target),
        "e_start_mwh": e_init, "e_end_mwh": float(s.e[-1]),
        "compute_time_s": plan.compute_time_s, "wall_time_s": wall,
        **(plan.diagnostics or {}),
    }


def run_backtest(policy: Policy, data: MarketData, days: list, resolution: str, battery: Battery,
                 market: Market, publication_hhmm: str = "13:00", progress: bool = False) -> pl.DataFrame:
    rows = []
    e_init = battery.e_init
    prev_day = None
    it = days
    if progress:
        from tqdm import tqdm
        it = tqdm(days, desc=policy.name)
    for day in it:
        day = cal.to_date(day)
        if not (policy.carries_soc and prev_day == day - dt.timedelta(days=1)):
            e_init = battery.e_init
        row = settle_day(policy, data, day, resolution, e_init, battery, market, publication_hhmm)
        rows.append(row)
        e_init = row["e_end_mwh"]
        prev_day = day
    return pl.DataFrame(rows)
